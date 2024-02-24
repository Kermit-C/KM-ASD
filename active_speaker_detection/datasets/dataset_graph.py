#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import math
import random
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data, Dataset

from active_speaker_detection.models.graph_layouts import (
    get_spatial_connection_pattern,
    get_temporal_connection_pattern,
)

from .data_store_emb import EmbeddingDataStore


class GraphDataset(Dataset):

    def __init__(
        self,
        embedding_root,
        graph_time_steps: int,  # 图的时间步数
        stride: int,  # 步长
        context_size,  # 上下文大小，即上下文中有多少个实体
    ):
        super().__init__()
        self.store = EmbeddingDataStore(embedding_root)

        # 图上下文
        self.context_size = context_size  # 上下文大小，即上下文中有多少个实体

        # 获取空间连接模式
        spatial_connection_pattern = get_spatial_connection_pattern(
            context_size, graph_time_steps
        )
        spatial_src_edges = spatial_connection_pattern["src"]
        spatial_dst_edges = spatial_connection_pattern["dst"]
        self.spatial_batch_edges = torch.tensor(
            [spatial_src_edges, spatial_dst_edges], dtype=torch.long
        )

        # 获取时间连接模式
        temporal_connection_pattern = get_temporal_connection_pattern(
            context_size, graph_time_steps
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
        return len(self.store.entity_list)

    def __getitem__(self, index):
        """根据 entity_list 的索引，获取数据集中的一个数据
        那么，数据集中单个数据的定义是，以单个实体为中心，上下文大小为上下文，取一个实体的时间上下文中，获取视频和音频特征，以及标签
        """
        video_id, entity_id = self.store.entity_list[index]
        target_entity_metadata = self.store.entity_data[video_id][entity_id]
        # 随机选择一个中间时间戳的索引
        # TODO: 这样是不是没把所有的时间戳都用上
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

            # 获取视频特征和标签
            video_data, target_v, entities_v = self.store.get_video_data(
                video_id,
                entity_id,
                tc,
                self.context_size,
            )
            # 获取音频特征和标签
            audio_data, target_a, entity_a = self.store.get_audio_data(
                video_id, entity_id, tc
            )

            # 填充标签
            target_set.append(target_a)
            for tv in target_v:
                target_set.append(tv)
            # 填充实体
            entities_set.append(entity_a)
            for ev in entities_v:
                entities_set.append(ev)

            # 创建一个 tensor，用于存放特征数据，这里是 2D 的，第一个维度是时刻上下文节点数*时间上下文数，第二个维度是特征维度
            # IMPORTANT: 节点所在是第一个维度
            video_data = [torch.from_numpy(v) for v in video_data]
            if feature_set is None:
                feature_set = torch.zeros(
                    nodes_per_time * (self.graph_time_steps),
                    video_data[0].size(0),
                )

            # 图节点的偏移，在 feature_set 中的偏移
            graph_offset = time_idx * nodes_per_time
            # 第一个维度的第一个节点是音频特征
            audio_data = torch.from_numpy(audio_data)
            feature_set[graph_offset] = audio_data
            # 填充视频特征
            for i in range(self.context_size):
                feature_set[graph_offset + (i + 1)] = video_data[i]

        return Data(
            x=feature_set,
            edge_index=(self.spatial_batch_edges, self.temporal_batch_edges),  # type: ignore
            y=torch.tensor(target_set),
        )
