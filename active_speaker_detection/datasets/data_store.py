#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import active_speaker_detection.utils.clip_utils as cu
import active_speaker_detection.utils.io_util as io
from active_speaker_detection.utils.file_util import (
    csv_to_list,
    postprocess_entity_label,
    postprocess_speech_label,
)


class DataStore:
    """数据存储"""

    """
    csv 标注格式：
    | video_id | frame_timestamp | entity_box_x1 | entity_box_y1 | entity_box_x2 | entity_box_y2 | label | entity_id | label_id | instance_id |
    """

    def __init__(self, audio_root, video_root, csv_file_path):
        # 数据根目录
        self.audio_root = audio_root  # 音频根目录
        self.video_root = video_root  # 视频根目录

        # 一个实体对应 video+entity

        # 一个人在什么时候是否说话 (entity_id, timestamp, entity_label)
        self.entity_data: Dict[str, Dict[str, List[Tuple[str, str, int]]]] = {}
        # 一个时间点是否说话，用于音频节点
        self.speech_data: Dict[str, Dict[str, int]] = {}
        # 一个时间点的实体们
        self.ts_to_entity: Dict[str, Dict[str, List[str]]] = {}

        # 所有实体的列表 (video_id, entity_id)
        self.entity_list: List[Tuple[str, str]] = []
        # 所有特征的列表 (video_id, entity_id, timestamp, entity_label)
        self.feature_list: List[Tuple[str, str, str, int]] = []

        # 读取并处理标签文件到 cache
        entity_set = self.cache_entity_data(csv_file_path)
        self.postprocess_entity_list(entity_set)

    def cache_entity_data(self, csv_file_path):
        """读取标签文件到 cache，返回所有实体的 set"""
        # 保存所有实体的 set, (video_id, entity_id)
        entity_set: set[Tuple[str, str]] = set()

        csv_data = csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        # csv_data 是一个二维列表，每一行是一个列表，每一行的元素是一个字符串
        for csv_row in csv_data:
            # TMP: 限制实体数量，Debug 用
            # if len(entity_set) > 50:
            #     break
            if len(csv_row) != 10:
                # 如果不是 10 个元素，就跳过
                continue
            video_id = csv_row[0]  # 视频 id
            entity_id = csv_row[-3]  # 实体 id
            timestamp = csv_row[1]  # 时间戳

            speech_label = postprocess_speech_label(csv_row[-2])
            entity_label = postprocess_entity_label(csv_row[-2])
            # 最小的实体数据，什么人在什么时间什么状态
            minimal_entity_data = (
                entity_id,
                timestamp,
                entity_label,
            )

            # 存储最少的实体数据
            # 先判断 video_id 是否在 entity_data 中，如果不在，就添加一个空字典
            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}
            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
                entity_set.add((video_id, entity_id))
            # 将 minimal_entity_data 添加到 entity_data 中
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

            # 存储语音元数据
            if video_id not in self.speech_data.keys():
                self.speech_data[video_id] = {}
            if timestamp not in self.speech_data[video_id].keys():
                self.speech_data[video_id][timestamp] = speech_label
            # 相当于聚合一个时间点所有人说话状态
            new_speech_label = max(self.speech_data[video_id][timestamp], speech_label)
            self.speech_data[video_id][timestamp] = new_speech_label

        return entity_set

    def postprocess_entity_list(self, entity_set):
        """后处理 _cache_entity_data 得到的数据"""
        print("Initial", len(entity_set))

        # 过滤掉磁盘上没有的实体，video_root 中有每一个实体的文件夹
        print("video_root", self.video_root)
        all_disk_data = set(os.listdir(self.video_root))  # type: ignore
        for video_id, entity_id in entity_set.copy():
            if entity_id.replace(":", "_") not in all_disk_data:
                entity_set.remove((video_id, entity_id))
        print("Pruned not in disk", len(entity_set))

        # 过滤掉实体文件夹中，timestamp 画面数和 entity_data 中的数量不一致的实体
        for video_id, entity_id in entity_set.copy():
            dir = os.path.join(self.video_root, entity_id.replace(":", "_"))  # type: ignore
            if len(os.listdir(dir)) != len(self.entity_data[video_id][entity_id]):
                entity_set.remove((video_id, entity_id))

        print("Pruned not complete", len(entity_set))
        self.entity_list = sorted(list(entity_set))

        for video_id, entity_id in entity_set:
            ent_min_data = self.entity_data[video_id][entity_id]

            # 保存时间点对应实体列表的关系字典
            if video_id not in self.ts_to_entity.keys():
                self.ts_to_entity[video_id] = {}
            for ed in ent_min_data:
                timestamp = ed[1]
                if timestamp not in self.ts_to_entity[video_id].keys():
                    self.ts_to_entity[video_id][timestamp] = []
                self.ts_to_entity[video_id][timestamp].append(entity_id)

            # 存储所有特征的列表
            for ed in ent_min_data:
                self.feature_list.append((video_id, entity_id, ed[1], ed[2]))

    def get_speaker_context(
        self, video_id: str, target_entity_id: str, center_ts: str, ctx_len: int
    ) -> List[str]:
        """
        获取同一时间点的上下文，返回一个列表，列表中的元素是实体 id
        :param video_id: 视频 id
        :param target_entity_id: 目标实体 id
        :param center_ts: 时间戳
        :param ctx_len: 上下文长度，实体数量
        """
        # 获取包含自己的上下文实体列表
        context_entities = list(self.ts_to_entity[video_id][center_ts])
        random.shuffle(context_entities)
        # 排除自己
        context_entities.remove(target_entity_id)

        if not context_entities:
            # 如果没有上下文，就返回自己
            context_entities.insert(0, target_entity_id)
            while len(context_entities) < ctx_len:  # self is context
                # 从 context_entities 中随机选择一个实体，添加到 context_entities 中
                # 补全到 ctx_len 的长度
                context_entities.append(random.choice(context_entities))
        elif len(context_entities) < ctx_len:
            # 保证自己在第一个位置
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < ctx_len:
                # 从 context_entities 中随机选择一个实体，添加到 context_entities 中
                # 补全到 ctx_len 的长度
                context_entities.append(random.choice(context_entities[1:]))
        else:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            # 太长了，就截断
            context_entities = context_entities[:ctx_len]

        return context_entities

    def get_time_context(
        self,
        entity_data: List[Tuple[str, str, int]],
        target_index: int,
        graph_time_steps: int,
        stride: int,
    ) -> List[str]:
        """获取时间上下文，返回时间戳列表"""
        # 所有时间戳
        all_ts = [ed[1] for ed in entity_data]
        # 中心时间戳，即目标时间戳
        center_ts = entity_data[target_index][1]
        # 中心时间戳的索引
        center_ts_idx = all_ts.index(str(center_ts))

        half_time_steps = int(graph_time_steps / 2)
        start = center_ts_idx - (half_time_steps * stride)
        end = center_ts_idx + ((half_time_steps + 1) * stride)
        # 选取的时间戳索引
        selected_ts_idx = list(range(start, end, stride))

        selected_ts = []
        for i, idx in enumerate(selected_ts_idx):
            # 保证不越界，时间上下文中都要当前所表示的实体，越界就取边界值
            if idx < 0:
                idx = 0
            if idx >= len(all_ts):
                idx = len(all_ts) - 1
            selected_ts.append(all_ts[idx])

        return selected_ts

    def search_ts_in_meta_data(
        self, entity_metadata: List[Tuple[str, str, int]], ts: str
    ):
        """在 entity_metadata 中搜索时间戳 ts，返回索引"""
        for idx, em in enumerate(entity_metadata):
            if em[1] == ts:
                return idx
        raise Exception("Bad Context")

    def get_audio_size(self, half_clip_length: int) -> Tuple[Tuple[int, int, int]]:
        """获得音频的大小，返回一个元组(1, T, W)"""
        video_id, entity_id = self.entity_list[0]
        # 一个人的所有时间戳的元数据
        entity_metadata = self.entity_data[video_id][entity_id]
        audio_offset = float(entity_metadata[0][1])
        # 随机选择一个中间时间戳的索引
        mid_index = random.randint(0, len(entity_metadata) - 1)

        # 生成一个长度为 half_clip_size*2+1 的单人时间片段
        clip_meta_data: List[Tuple[str, str, int]] = cu.generate_clip_meta(
            entity_metadata, mid_index, half_clip_length
        )
        # 从片段元数据中获得音频梅尔特征
        audio_data, audio_fbank = io.load_a_clip_from_metadata(
            clip_meta_data, self.video_root, self.audio_root, audio_offset
        )
        return np.float32(audio_data).shape, np.float32(audio_fbank).shape  # type: ignore

    def get_audio_data(
        self, video_id: str, entity_id: str, mid_index: int, half_clip_length: int
    ) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """根据实体 ID 和某刻时间戳索引，获取音频梅尔特征(1, T, W)、fbank(1, 80, T)和标签"""
        entity_metadata = self.entity_data[video_id][entity_id]
        # 获取这个实体的音频偏移
        audio_offset = float(entity_metadata[0][1])
        midone: Tuple[str, str, int] = entity_metadata[mid_index]
        # 获取目标时刻是否说话
        target_audio = self.speech_data[video_id][midone[1]]
        # 获取目标时刻谁说话，计算 md5 作为 int 编号
        target_entity = ""
        if target_audio == 1:
            # 获取包含自己的上下文实体列表
            mid_ts = self.entity_data[video_id][entity_id][mid_index][1]
            context_entities = list(self.ts_to_entity[video_id][mid_ts])
            for ctx_entity in context_entities:
                # 看哪个实体在这个时间点说话
                ctx_entity_metadata = self.entity_data[video_id][ctx_entity]
                ctx_mid_index = self.search_ts_in_meta_data(ctx_entity_metadata, mid_ts)
                if self.entity_data[video_id][ctx_entity][ctx_mid_index][-1] == 1:
                    target_entity = ctx_entity
                    break

        # 生成一个长度为 half_clip_size*2+1 的单人时间片段
        clip_meta_data = cu.generate_clip_meta(
            entity_metadata, mid_index, half_clip_length
        )
        # 从片段元数据中获得音频梅尔特征
        audio_data, audio_fbank = io.load_a_clip_from_metadata(
            clip_meta_data, self.video_root, self.audio_root, audio_offset
        )
        return (
            np.float32(audio_data),
            np.float32(audio_fbank),
            target_audio,
            target_entity,
        )  # type: ignore

    def get_video_data(
        self,
        video_id: str,
        entity_id: str,
        mid_index: int,
        context_size: int,
        half_clip_length: int,
        video_transform: Callable,
        video_augment: Optional[Callable] = None,
        video_augment_crop_ratio: Optional[float] = None,
        cache: Optional[dict] = None,
    ) -> Tuple[List[List[torch.Tensor]], List[int], List[str]]:
        """根据实体 ID 和中间某刻时间戳索引，获取所有人视频画面 List(人数 list(clip数)) tensor(通道数 * 高度 * 宽度) 和这一刻所有实体的标签
        :param video_id: 视频 id
        :param entity_id: 实体 id
        :param mid_index: 时间戳索引，即中间时间戳的索引
        """
        orginal_entity_metadata = self.entity_data[video_id][entity_id]
        # 获取中间时间戳
        time_ent = orginal_entity_metadata[mid_index][1]
        # 获取上下文的实体 id list
        context = self.get_speaker_context(video_id, entity_id, time_ent, context_size)

        # 视频数据，同一个人的是 List 的一个元素
        video_data: List[List[Image.Image]] = []
        # 标签，索引和 video_data 对应
        targets: list[int] = []
        # 实体
        entities: list[str] = []
        for ctx_entity in context:
            # 查找时间戳在 entity_metadata 中的索引
            entity_metadata = self.entity_data[video_id][ctx_entity]
            # 获取中间时间戳在 entity_metadata 中的索引
            ts_idx = self.search_ts_in_meta_data(entity_metadata, time_ent)
            # 获取到标签
            target_ctx = int(entity_metadata[ts_idx][-1])

            # 生成一个长度为 half_clip_size*2+1 的单人时间片段
            clip_meta_data = cu.generate_clip_meta(
                entity_metadata, ts_idx, half_clip_length
            )
            # 从片段元数据中获得视频 Image list
            video_data.append(
                io.load_v_clip_from_metadata_cache(
                    clip_meta_data, self.video_root, cache
                )
                if cache is not None
                else io.load_v_clip_from_metadata(clip_meta_data, self.video_root)
            )
            targets.append(target_ctx)
            entities.append(ctx_entity)

        if video_augment is not None:
            # 视频增强
            video_data = [
                video_augment(vd, video_augment_crop_ratio) for vd in video_data
            ]

        # 视频转换 tensor
        for vd_idx, vd in enumerate(video_data):
            tensor_vd = [video_transform(f) for f in vd]
            video_data[vd_idx] = tensor_vd  # type: ignore

        return video_data, targets, entities  # type: ignore

    def parse_entities_to_int(self, entities: List[str]) -> List[int]:
        """将实体列表转换为 int"""
        entities_set: set[str] = set(entities)
        entities_list: list[str] = list(entities_set)
        return [entities_list.index(e) for e in entities]
