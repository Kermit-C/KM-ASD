#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import random
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data, Dataset

from .data_store_emb import EmbeddingDataStore


class GraphDataset(Dataset):

    def __init__(
        self,
        embedding_root,
        data_store_cache,
        graph_time_steps: int,  # 图的时间步数
        graph_time_stride: int,  # 步长
        is_edge_double=False,  # 是否边是双向的
        is_edge_across_entity=False,  # 是否跨实体连接，即不同实体不同时刻之间的连接
        max_context: Optional[int] = None,  # 最大上下文大小，即上下文中有多少个实体
    ):
        super().__init__()
        self.store = EmbeddingDataStore(embedding_root, data_store_cache)

        # 时序层配置
        self.graph_time_steps = graph_time_steps  # 图的时间步数
        self.graph_time_stride = graph_time_stride  # 图的步长
        self.is_edge_double = is_edge_double  # 是否边是双向的
        self.is_edge_across_entity = is_edge_across_entity

        # 图上下文
        self.max_context = max_context  # 上下文大小，即上下文中有多少个实体

    def __len__(self):
        return len(self.store.entity_list)

    def __getitem__(self, index):
        """数据集中单个数据的定义是，以单个实体为中心，取实体时间线上所有相关实体的特征和标签，构成一个图"""
        video_id, entity_id = self.store.entity_list[index]
        target_entity_metadata = self.store.entity_data[video_id][entity_id]
        # 随机选择一个中间时间戳的索引
        center_index = random.randint(0, len(target_entity_metadata) - 1)
        # 获取时间上下文，所有时间戳列表
        time_context: List[str] = self.store.get_time_context(
            target_entity_metadata,
            center_index,
            0,  # 原来是 self.graph_time_steps，这是为了形成一整个大图
            self.graph_time_stride,
        )

        # 图节点特征
        feature_list: list[torch.Tensor] = []
        # 图节点的标签数据
        target_list: list[int] = []
        # 图节点的实体数据
        entity_list: list[str] = []
        # 时间戳列表
        timestamp_list: list[str] = []
        # 位置列表
        position_list: list[Tuple[float, float, float, float]] = []

        # 对每个上下文时间戳，获取视频特征和标签
        cache = {}
        for time_idx, timestamp in enumerate(time_context):
            # 获取视频特征和标签
            video_data, target_v, entities_v, positions = self.store.get_video_data(
                video_id, entity_id, timestamp, self.max_context, cache
            )
            # 获取音频特征和标签
            audio_data, target_a, entity_a = self.store.get_audio_data(
                video_id, entity_id, timestamp, cache
            )

            a_feat = torch.from_numpy(audio_data)
            for v_np, target, entity, pos in zip(
                video_data, target_v, entities_v, positions
            ):
                v_feat = torch.from_numpy(v_np)
                # 一个特征是 2 * 128 维的，音频特征和视频特征
                feature_list.append(torch.stack([a_feat, v_feat], dim=0))
                target_list.append(target)
                entity_list.append(entity)
                timestamp_list.append(timestamp)
                position_list.append(pos)

        # 边的出发点，每一条无向边会正反记录两次
        source_vertices: list[int] = []
        # 边的结束点，每一条无向边会正反记录两次
        target_vertices: list[int] = []
        # 边出发点的位置信息，x1, y1, x2, y2
        source_vertices_pos: list[Tuple[float, float, float, float]] = []
        # 边结束点的位置信息，x1, y1, x2, y2
        target_vertices_pos: list[Tuple[float, float, float, float]] = []

        # 构造边
        for i, (entity, timestamp) in enumerate(zip(entity_list, timestamp_list)):
            for j, (entity, timestamp) in enumerate(zip(entity_list, timestamp_list)):
                # 自己不连接自己
                if i == j:
                    continue

                # 超过了时间步数，不连接
                if abs(i - j) > self.graph_time_steps:
                    continue

                # 只单向连接
                if not self.is_edge_double and i > j:
                    continue

                if timestamp_list[i] == timestamp_list[j]:
                    # 同一时刻上下文中的实体之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                elif entity_list[i] == entity_list[j]:
                    # 同一实体在不同时刻之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                elif self.is_edge_across_entity:
                    # 不同实体在不同时刻之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])

        return Data(
            # 维度为 [节点数量, 2, 128]，表示每个节点的音频和视频特征
            x=torch.stack(feature_list, dim=0),
            # 维度为 [2, 边的数量]，表示每条边的两侧节点的索引
            edge_index=torch.tensor(
                [source_vertices, target_vertices], dtype=torch.long
            ),
            # 维度为 [边的数量, 2, 4]，表示每条边的两侧节点的位置信息
            edge_attr=torch.tensor(
                [source_vertices_pos, target_vertices_pos], dtype=torch.float
            ).transpose(0, 1),
            # 维度为 [节点数量]，表示每个节点的标签
            y=torch.tensor(target_list),
        )
