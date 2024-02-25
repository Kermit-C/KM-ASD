#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import os
import pickle
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import active_speaker_detection.utils.clip_utils as cu
import active_speaker_detection.utils.io_util as io


class DataStore:
    """数据存储"""

    def __init__(self, audio_root, video_root, data_store_cache):
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

        # 读取 Cache
        self.load_cache(data_store_cache)

    def load_cache(self, data_store_cache: str):
        """加载缓存"""
        if os.path.exists(data_store_cache):
            with open(data_store_cache, "rb") as f:
                cache = pickle.load(f)
                self.entity_data = cache["entity_data"]
                self.speech_data = cache["speech_data"]
                self.ts_to_entity = cache["ts_to_entity"]
                self.entity_list = cache["entity_list"]
                self.feature_list = cache["feature_list"]

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
        """获得音频的大小，返回一个元组(1, 13, T)"""
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
        """根据实体 ID 和某刻时间戳索引，获取音频梅尔特征(1, 13, T)、fbank(1, 80, T)和标签"""
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
