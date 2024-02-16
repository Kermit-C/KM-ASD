#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Data, Dataset
from torchvision import transforms

import active_speaker_detection.utils.clip_utils as cu
import active_speaker_detection.utils.io_util as io
from active_speaker_detection.utils.augmentations import (
    video_corner_crop,
    video_temporal_crop,
)
from active_speaker_detection.utils.file_util import (
    csv_to_list,
    postprocess_entity_label,
    postprocess_speech_label,
)
from active_speaker_detection.utils.hash_util import calculate_md5

"""
csv 标注格式：
| video_id | frame_timestamp | entity_box_x1 | entity_box_y1 | entity_box_x2 | entity_box_y2 | label | entity_id | label_id | instance_id |
"""


class GraphContextualDataset(Dataset):
    """
    图上下文基本类，只用来被继承
    """

    def __init__(self):
        # 一个实体对应 video+entity

        # 一个人在什么时候是否说话 (entity_id, timestamp, entity_label)
        self.entity_data: Dict[str, Dict[str, List[Tuple[str, int, int]]]] = {}
        # 一个时间点是否说话，用于音频节点
        self.speech_data: Dict[str, Dict[int, int]] = {}
        # 一个时间点的实体们
        self.ts_to_entity: Dict[str, Dict[int, List[str]]] = {}

        # 所有实体的列表 (video_id, entity_id)
        self.entity_list: List[Tuple[str, str]] = []

    def get_speaker_context(
        self, video_id: str, target_entity_id: str, center_ts: int, ctx_len: int
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

    def search_ts_in_meta_data(
        self, entity_metadata: List[Tuple[str, int, int]], ts: int
    ):
        """在 entity_metadata 中搜索时间戳 ts，返回索引"""
        for idx, em in enumerate(entity_metadata):
            if em[1] == ts:
                return idx
        raise Exception("Bad Context")

    def _cache_entity_data(self, csv_file_path):
        """读取标签文件到 cache，返回所有实体的 set"""
        # 保存所有实体的 set, (video_id, entity_id)
        entity_set: set[Tuple[str, str]] = set()

        csv_data = csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        # csv_data 是一个二维列表，每一行是一个列表，每一行的元素是一个字符串
        for csv_row in csv_data:
            # TEMP: 限制实体数量，Debug 用
            if len(entity_set) > 50:
                break
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

    def _entity_list_postprocessing(self, entity_set):
        """后处理 _cache_entity_data 得到的数据"""
        print("Initial", len(entity_set))

        # 过滤掉磁盘上没有的实体，video_root 中有每一个实体的文件夹
        print("video_root", self.video_root)
        all_disk_data = set(os.listdir(self.video_root))
        for video_id, entity_id in entity_set.copy():
            if entity_id.replace(":", "_") not in all_disk_data:
                entity_set.remove((video_id, entity_id))
        print("Pruned not in disk", len(entity_set))

        # 过滤掉实体文件夹中，timestamp 画面数和 entity_data 中的数量不一致的实体
        for video_id, entity_id in entity_set.copy():
            dir = os.path.join(self.video_root, entity_id.replace(":", "_"))
            if len(os.listdir(dir)) != len(self.entity_data[video_id][entity_id]):
                entity_set.remove((video_id, entity_id))

        print("Pruned not complete", len(entity_set))
        self.entity_list = sorted(list(entity_set))

        # Allocate Simultanous Entities
        # 保存时间点对应实体列表的关系字典
        for video_id, entity_id in entity_set:
            if video_id not in self.ts_to_entity.keys():
                self.ts_to_entity[video_id] = {}

            ent_min_data = self.entity_data[video_id][entity_id]
            for ed in ent_min_data:
                timestamp = ed[1]
                if timestamp not in self.ts_to_entity[video_id].keys():
                    self.ts_to_entity[video_id][timestamp] = []
                self.ts_to_entity[video_id][timestamp].append(entity_id)


class GraphDatasetETE(GraphContextualDataset):
    """
    图数据集基本类，只用来被继承
    """

    def __init__(
        self,
        audio_root,
        video_root,
        csv_file_path,
        context_size,  # 上下文大小，即上下文中有多少个实体
        clip_lenght,  # 片段长度，短时序上下文片段的长度
        connection_pattern: Dict[
            str, List[int]
        ],  # 一个字典，包含 src 和 dst 两个键，对应的值是一个列表，表示边的连接关系
        video_transform: transforms.Compose = None,  # 视频转换方法
        do_video_augment=False,  # 是否视频增强
        crop_ratio=0.8,  # 视频裁剪比例
        norm_audio=False,  # 是否归一化音频
    ):
        super().__init__()

        # 数据根目录
        self.audio_root = audio_root  # 音频根目录
        self.video_root = video_root  # 视频根目录

        # 后处理
        self.crop_ratio = crop_ratio
        self.video_transform = (
            video_transform if video_transform is not None else transforms.ToTensor()
        )
        self.do_video_augment = do_video_augment  # 是否视频增强

        # 图上下文
        self.context_size = context_size  # 上下文大小，即上下文中有多少个实体

        # 图节点的配置
        self.norm_audio = norm_audio  # 是否归一化音频
        self.half_clip_length = math.floor(clip_lenght / 2)  # 一半的片段长度

        # 读取并处理标签文件到 cache
        entity_set = self._cache_entity_data(csv_file_path)
        self._entity_list_postprocessing(entity_set)

        # 边的连接关系
        src_edges = connection_pattern["src"]
        dst_edges = connection_pattern["dst"]
        self.batch_edges = torch.tensor([src_edges, dst_edges], dtype=torch.long)

        # 复制一份 entity_list
        self.entity_list.extend(self.entity_list)

    def __len__(self):
        return int(len(self.entity_list) / 1)

    def get_audio_size(
        self,
    ) -> Tuple[Tuple[int, int, int]]:
        """获得音频的大小，返回一个元组(1, 13, T)"""
        video_id, entity_id = self.entity_list[0]
        # 一个人的所有时间戳的元数据
        entity_metadata = self.entity_data[video_id][entity_id]
        audio_offset = float(entity_metadata[0][1])
        # 随机选择一个中间时间戳的索引
        mid_index = random.randint(0, len(entity_metadata) - 1)

        # 生成一个长度为 half_clip_size*2+1 的单人时间片段
        clip_meta_data: List[Tuple[str, int, int]] = cu.generate_clip_meta(
            entity_metadata, mid_index, self.half_clip_length
        )
        # 从片段元数据中获得音频梅尔特征
        audio_data, audio_fbank = io.load_a_clip_from_metadata(
            clip_meta_data, self.video_root, self.audio_root, audio_offset
        )
        return np.float32(audio_data).shape, np.float32(audio_fbank).shape

    def _get_scene_video_data(
        self, video_id: str, entity_id: str, mid_index: int
    ) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """根据实体 ID 和中间某刻时间戳索引，获取所有人视频画面 List(人数) tensor((clip数 * 通道数) * 高度 * 宽度) 和这一刻所有实体的标签
        :param video_id: 视频 id
        :param entity_id: 实体 id
        :param mid_index: 时间戳索引，即中间时间戳的索引
        """
        orginal_entity_metadata = self.entity_data[video_id][entity_id]
        # 获取中间时间戳
        time_ent = orginal_entity_metadata[mid_index][1]
        # 获取上下文的实体 id list
        context: List[str] = self.get_speaker_context(
            video_id, entity_id, time_ent, self.context_size
        )

        # 视频数据，同一个人的是 List 的一个元素
        video_data: List[List[Image.Image]] = []
        # 标签，索引和 video_data 对应
        targets: List[int] = []
        # 实体
        entities: List[int] = []
        for ctx_entity in context:
            # 查找时间戳在 entity_metadata 中的索引
            entity_metadata = self.entity_data[video_id][ctx_entity]
            # 获取中间时间戳在 entity_metadata 中的索引
            ts_idx = self.search_ts_in_meta_data(entity_metadata, time_ent)
            # 获取到标签
            target_ctx = int(entity_metadata[ts_idx][-1])

            # 生成一个长度为 half_clip_size*2+1 的单人时间片段
            clip_meta_data = cu.generate_clip_meta(
                entity_metadata, ts_idx, self.half_clip_length
            )
            # 从片段元数据中获得视频 Image list
            video_data.append(
                io.load_v_clip_from_metadata(clip_meta_data, self.video_root)
            )
            targets.append(target_ctx)
            entities.append(calculate_md5(ctx_entity))

        if self.do_video_augment:
            # 随机视频增强
            video_data = [video_temporal_crop(vd, self.crop_ratio) for vd in video_data]

        # 视频转换 tensor
        for vd_idx, vd in enumerate(video_data):
            tensor_vd = [self.video_transform(f) for f in vd]
            video_data[vd_idx] = tensor_vd

        # 把一个人的多个画面在通道维度上合并成一个 3D tensor
        video_data = [torch.cat(vd, dim=0) for vd in video_data]
        return video_data, targets, entities

    def _get_audio_data(
        self, video_id: str, entity_id: str, mid_index: int
    ) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """根据实体 ID 和某刻时间戳索引，获取音频梅尔特征(1, 13, T)、fbank(1, 80, T)和标签"""
        entity_metadata = self.entity_data[video_id][entity_id]
        # 获取这个实体的音频偏移
        audio_offset = float(entity_metadata[0][1])
        midone: Tuple[str, int, int] = entity_metadata[mid_index]
        # 获取目标时刻是否说话
        target_audio = self.speech_data[video_id][midone[1]]
        # 获取目标时刻谁说话，计算 md5 作为 int 编号
        target_entity = 0
        if target_audio == 1:
            # 获取包含自己的上下文实体列表
            mid_ts = self.entity_data[video_id][entity_id][mid_index][1]
            context_entities = list(self.ts_to_entity[video_id][mid_ts])
            for ctx_entity in context_entities:
                # 看哪个实体在这个时间点说话
                ctx_entity_metadata = self.entity_data[video_id][ctx_entity]
                ctx_mid_index = self.search_ts_in_meta_data(ctx_entity_metadata, mid_ts)
                if self.entity_data[video_id][ctx_entity][ctx_mid_index][-1] == 1:
                    target_entity = calculate_md5(ctx_entity)
                    break

        # 生成一个长度为 half_clip_size*2+1 的单人时间片段
        clip_meta_data = cu.generate_clip_meta(
            entity_metadata, mid_index, self.half_clip_length
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
        )

    def __getitem__(self, index):
        """根据 entity_list 的索引，获取数据集中的一个数据
        那么，数据集中单个数据的定义是，以单个实体为中心，上下文大小为上下文，中心时间戳为中心，获取视频和音频特征，以及标签
        从而形成一个特定时间点的小图
        """
        video_id, entity_id = self.entity_list[index]
        target_entity_metadata = self.entity_data[video_id][entity_id]
        # 随机选择一个中间时间戳的索引
        target_index = random.randint(0, len(target_entity_metadata) - 1)

        # 获取视频特征和标签
        video_data, target_v, entities_v = self._get_scene_video_data(
            video_id, entity_id, target_index
        )
        # 获取音频特征和标签
        audio_data, audio_fbank, target_a, entity_a = self._get_audio_data(
            video_id, entity_id, target_index
        )

        if self.norm_audio:
            # 归一化音频，这里是 z-score 归一化
            audio_data = (audio_data + 3.777757875102366) / 186.4988690376491

        # 填充标签
        target_set: List[int] = []
        target_set.append(target_a)
        for tv in target_v:
            target_set.append(tv)
        # 填充实体
        entities_set: List[int] = []
        entities_set.append(entity_a)
        for ev in entities_v:
            entities_set.append(ev)

        # 创建一个 tensor，用于存放特征数据，这里是 4D 的，第一个维度是上下文大小+1，第二个维度是(clip数*通道数)，第三四个维度是画面的长宽
        feature_set = torch.zeros(
            (
                len(video_data) + 1,
                video_data[0].size(0),
                video_data[0].size(1),
                video_data[0].size(2),
            )
        )
        audio_fbank = torch.from_numpy(audio_fbank)
        vfal_feature_set = torch.zeros((1, audio_fbank.size(1), audio_fbank.size(2)))
        # 音频特征数据，第一个维度是 1，第二个维度是梅尔特征的窗口数，第三个维度是时间
        # 第一维的第一个元素是音频特征
        audio_data = torch.from_numpy(audio_data)
        feature_set[0, 0, : audio_data.size(1), : audio_data.size(2)] = audio_data
        # 填充视频特征
        for i in range(self.context_size):
            feature_set[i + 1, ...] = video_data[i]
            vfal_feature_set[i + 1, ...] = audio_fbank

        vfal_feature_set[0, : audio_fbank.size(1), : audio_fbank.size(2)] = audio_fbank

        return Data(
            x=feature_set,
            x2=vfal_feature_set,
            edge_index=self.batch_edges,
            y=torch.tensor(target_set),
            y2=torch.tensor([e % int(math.pow(2, 31)) for e in entities_set]),
        )


class IndependentGraphDatasetETE3D(GraphDatasetETE):

    def __init__(
        self,
        audio_root,
        video_root,
        csv_file_path,
        graph_time_steps: int,  # 图的时间步数
        stride: int,  # 步长
        context_size,
        clip_lenght,
        spatial_connection_pattern,
        temporal_connection_pattern,
        video_transform: transforms.Compose = None,
        do_video_augment=False,
        crop_ratio=0.95,
        norm_audio=False,
    ):
        super().__init__(
            audio_root,
            video_root,
            csv_file_path,
            context_size,
            clip_lenght,
            spatial_connection_pattern,
            video_transform,
            do_video_augment,
            crop_ratio,
            norm_audio,
        )

        # 父类的 batch_edges
        self.batch_edges = None

        spatial_src_edges = spatial_connection_pattern["src"]
        spatial_dst_edges = spatial_connection_pattern["dst"]
        self.spatial_batch_edges = torch.tensor(
            [spatial_src_edges, spatial_dst_edges], dtype=torch.long
        )

        temporal_src_edges = temporal_connection_pattern["src"]
        temporal_dst_edges = temporal_connection_pattern["dst"]
        self.temporal_batch_edges = torch.tensor(
            [temporal_src_edges, temporal_dst_edges], dtype=torch.long
        )

        # 时序层配置
        self.graph_time_steps = graph_time_steps  # 图的时间步数
        self.stride = stride  # 图的步长

    def _get_scene_video_data(
        self, video_id: str, entity_id: str, mid_index: int, cache: dict
    ) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """父类方案的改进，加了 cache
        根据实体 ID 和中间某刻时间戳索引，获取所有人视频画面 List(人数) tensor(通道数 * clip数 * 高度 * 宽度) 和这一刻所有实体的标签
        """
        orginal_entity_metadata = self.entity_data[video_id][entity_id]
        # 获取中间时间戳
        time_ent = orginal_entity_metadata[mid_index][1]
        # 获取上下文的实体 id list
        context = self.get_speaker_context(
            video_id, entity_id, time_ent, self.context_size
        )

        # 视频数据，同一个人的是 List 的一个元素
        video_data: List[List[Image.Image]] = []
        # 标签，索引和 video_data 对应
        targets: list[int] = []
        # 实体
        entities: list[int] = []
        for ctx_entity in context:
            # 查找时间戳在 entity_metadata 中的索引
            entity_metadata = self.entity_data[video_id][ctx_entity]
            # 获取中间时间戳在 entity_metadata 中的索引
            ts_idx = self.search_ts_in_meta_data(entity_metadata, time_ent)
            # 获取到标签
            target_ctx = int(entity_metadata[ts_idx][-1])

            # 生成一个长度为 half_clip_size*2+1 的单人时间片段
            clip_meta_data = cu.generate_clip_meta(
                entity_metadata, ts_idx, self.half_clip_length
            )
            # 从片段元数据中获得视频 Image list
            video_data.append(
                io.load_v_clip_from_metadata_cache(
                    clip_meta_data, self.video_root, cache
                )
            )
            targets.append(target_ctx)
            entities.append(calculate_md5(ctx_entity))

        if self.do_video_augment:
            # 视频角落增强，但这里没给具体实现
            video_data = [video_corner_crop(vd, self.crop_ratio) for vd in video_data]

        # 视频转换 tensor
        for vd_idx, vd in enumerate(video_data):
            tensor_vd = [self.video_transform(f) for f in vd]
            video_data[vd_idx] = tensor_vd

        # 把一个人的多个画面合并成一个 4D tensor，这里和父类不一样
        video_data = [torch.stack(vd, dim=1) for vd in video_data]
        return video_data, targets, entities

    def _get_time_context(
        self, entity_data: List[Tuple[str, int, int]], target_index: int
    ) -> List[int]:
        """获取时间上下文，返回时间戳列表"""
        # 所有时间戳
        all_ts = [ed[1] for ed in entity_data]
        # 中心时间戳，即目标时间戳
        center_ts = entity_data[target_index][1]
        # 中心时间戳的索引
        center_ts_idx = all_ts.index(str(center_ts))

        half_time_steps = int(self.graph_time_steps / 2)
        start = center_ts_idx - (half_time_steps * self.stride)
        end = center_ts_idx + ((half_time_steps + 1) * self.stride)
        # 选取的时间戳索引
        selected_ts_idx = list(range(start, end, self.stride))

        selected_ts = []
        for i, idx in enumerate(selected_ts_idx):
            # 保证不越界，时间上下文中都要当前所表示的实体，越界就取边界值
            if idx < 0:
                idx = 0
            if idx >= len(all_ts):
                idx = len(all_ts) - 1
            selected_ts.append(all_ts[idx])

        return selected_ts

    def __getitem__(self, index):
        """根据 entity_list 的索引，获取数据集中的一个数据
        那么，数据集中单个数据的定义是，以单个实体为中心，上下文大小为上下文，取一个实体的时间上下文中，获取视频和音频特征，以及标签
        """
        video_id, entity_id = self.entity_list[index]
        target_entity_metadata = self.entity_data[video_id][entity_id]
        # 随机选择一个中间时间戳的索引
        center_index = random.randint(0, len(target_entity_metadata) - 1)
        # 获取时间上下文，时间戳列表
        time_context: List[int] = self._get_time_context(
            target_entity_metadata, center_index
        )

        # 图节点的特征数据
        feature_set: torch.Tensor = None
        # vfal 特征数据
        vfal_feature_set: torch.Tensor = None
        # 图节点的标签数据
        target_set: List[int] = []
        # 图节点的实体数据
        entities_set: List[int] = []

        # 所有时间戳
        all_ts = [ted[1] for ted in target_entity_metadata]
        # 时刻的上下文节点数，是上下文大小+1
        nodes_per_time = self.context_size + 1
        cache = {}
        for time_idx, tc in enumerate(time_context):
            # 对每个上下文时间戳，获取视频特征和标签

            target_index = all_ts.index(tc)

            # 获取视频特征和标签
            video_data, target_v, entities_v = self._get_scene_video_data(
                video_id, entity_id, target_index, cache
            )
            # 获取音频特征和标签
            audio_data, audio_fbank, target_a, entity_a = self._get_audio_data(
                video_id, entity_id, target_index
            )

            # 填充标签
            target_set.append(target_a)
            for tv in target_v:
                target_set.append(tv)
            # 填充实体
            entities_set.append(entity_a)
            for ev in entities_v:
                entities_set.append(ev)

            # 创建一个 tensor，用于存放特征数据，这里是 5D 的，第一个维度是时刻上下文节点数*时间上下文数，第二个维度是通道数，第三个维度是clip数，第四五个维度是画面的长宽
            # IMPORTANT: 节点所在是第一个维度
            if feature_set is None:
                feature_set = torch.zeros(
                    nodes_per_time * (self.graph_time_steps),
                    video_data[0].size(0),
                    video_data[0].size(1),
                    video_data[0].size(2),
                    video_data[0].size(3),
                )
            audio_fbank = torch.from_numpy(audio_fbank)
            if vfal_feature_set is None:
                vfal_feature_set = torch.zeros(
                    self.graph_time_steps,
                    audio_fbank.size(1),
                    audio_fbank.size(2),
                )

            # 图节点的偏移，在 feature_set 中的偏移
            graph_offset = time_idx * nodes_per_time
            # 第一个维度的第一个节点是音频特征
            audio_data = torch.from_numpy(audio_data)
            feature_set[
                graph_offset, 0, 0, : audio_data.size(1), : audio_data.size(2)
            ] = audio_data
            # 填充视频特征
            for i in range(self.context_size):
                feature_set[graph_offset + (i + 1), ...] = video_data[i]

            vfal_feature_set[time_idx, : audio_fbank.size(1), : audio_fbank.size(2)] = (
                audio_fbank
            )

        return Data(
            x=feature_set,
            x2=vfal_feature_set,
            edge_index=(self.spatial_batch_edges, self.temporal_batch_edges),
            y=torch.tensor(target_set),
            y2=torch.tensor([e % int(math.pow(2, 31)) for e in entities_set]),
        )
