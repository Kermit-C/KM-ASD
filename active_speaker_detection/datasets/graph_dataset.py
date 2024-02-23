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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Data, Dataset
from torchvision import transforms

from active_speaker_detection.utils.augmentations import video_temporal_crop

from .data_store import DataStore


class GraphDataset(Dataset):

    def __init__(
        self,
        audio_root,
        video_root,
        csv_file_path,
        graph_time_steps: int,  # 图的时间步数
        stride: int,  # 步长
        context_size,  # 上下文大小，即上下文中有多少个实体
        clip_lenght,  # 片段长度，短时序上下文片段的长度
        spatial_connection_pattern,
        temporal_connection_pattern,
        video_transform: Optional[transforms.Compose] = None,  # 视频转换方法
        do_video_augment=False,  # 是否视频增强
        crop_ratio=0.95,  # 视频裁剪比例
        norm_audio=False,  # 是否归一化音频
    ):
        super().__init__()
        self.store = DataStore(audio_root, video_root, csv_file_path)

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
        self.half_clip_length = math.floor(
            clip_lenght / 2
        )  # 每刻计算特征的帧数一半的长度

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

    def __len__(self):
        return int(len(self.store.entity_list) / 1)

    def __getitem__(self, index):
        """根据 entity_list 的索引，获取数据集中的一个数据
        那么，数据集中单个数据的定义是，以单个实体为中心，上下文大小为上下文，取一个实体的时间上下文中，获取视频和音频特征，以及标签
        """
        video_id, entity_id = self.store.entity_list[index]
        target_entity_metadata = self.store.entity_data[video_id][entity_id]
        # 随机选择一个中间时间戳的索引
        center_index = random.randint(0, len(target_entity_metadata) - 1)
        # 获取时间上下文，时间戳列表
        time_context: List[str] = self.store.get_time_context(
            target_entity_metadata, center_index, self.graph_time_steps, self.stride
        )

        # 图节点的特征数据
        feature_set: Optional[torch.Tensor] = None
        # vfal 特征数据
        vfal_feature_set: Optional[torch.Tensor] = None
        # 图节点的标签数据
        target_set: List[int] = []
        # 图节点的实体数据
        entities_set: List[str] = []

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
            audio_data, audio_fbank, target_a, entity_a = self.store.get_audio_data(
                video_id, entity_id, target_index, self.half_clip_length
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
            edge_index=(self.spatial_batch_edges, self.temporal_batch_edges),  # type: ignore
            y=torch.tensor(target_set),
            y2=torch.tensor(self.store.parse_entities_to_int(entities_set)),
        )

    def get_audio_size(
        self,
    ) -> Tuple[Tuple[int, int, int]]:
        """获得音频的大小，返回一个元组(1, 13, T)"""
        return self.store.get_audio_size(self.half_clip_length)

    def _get_scene_video_data(
        self, video_id: str, entity_id: str, mid_index: int, cache: dict
    ) -> Tuple[List[torch.Tensor], List[int], List[str]]:
        """根据实体 ID 和中间某刻时间戳索引
        获取所有人视频画面 List(人数) tensor(通道数 * clip数 * 高度 * 宽度) 和这一刻所有实体的标签
        :param video_id: 视频 id
        :param entity_id: 实体 id
        :param mid_index: 时间戳索引，即中间时间戳的索引
        :param cache: 缓存
        """
        video_data, targets, entities = self.store.get_video_data(
            video_id,
            entity_id,
            mid_index,
            self.context_size,
            self.half_clip_length,
            self.video_transform,
            video_temporal_crop if self.do_video_augment else None,
            self.crop_ratio if self.do_video_augment else None,
            cache,
        )
        # 把一个人的多个画面合并成一个 4D tensor
        return [torch.stack(vd, dim=1) for vd in video_data], targets, entities
