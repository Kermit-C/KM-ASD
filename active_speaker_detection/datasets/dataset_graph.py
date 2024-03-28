#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

from .data_store_feat import FeatureDataStore


class GraphDataset(Dataset):

    def __init__(
        self,
        feature_root,
        data_store_cache,
        graph_time_steps: int,  # 图的时间步数
        graph_time_stride: int,  # 步长
        graph_time_num: int,  # 多少个小图的时间步数构成一个大图
        is_edge_double=False,  # 是否边是双向的
        is_edge_across_entity=False,  # 是否跨实体连接，即不同实体不同时刻之间的连接
        max_context: int = 3,  # 最大上下文大小，即上下文中有多少个实体
    ):
        super().__init__()
        self.store = FeatureDataStore(feature_root, data_store_cache)

        # 时序层配置
        self.graph_time_steps = graph_time_steps  # 图的时间步数
        self.graph_time_stride = graph_time_stride  # 图的步长
        self.graph_time_num = graph_time_num  # 多少个小图的时间步数构成一个大图
        self.is_edge_double = is_edge_double  # 是否边是双向的
        self.is_edge_across_entity = is_edge_across_entity

        # 图上下文
        self.max_context = max_context  # 上下文大小，即上下文中有多少个实体

        self.cache = {}

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
            self.graph_time_steps * self.graph_time_num,
            self.graph_time_stride,
        )

        # 图节点特征
        feature_list: list[torch.Tensor] = []
        # 图节点的标签数据
        target_list: list[int] = []
        # 纯音频图节点掩码
        audio_feature_mask: list[bool] = []
        # 节点的时刻对应纯音频节点的索引
        audio_feature_idx_list: list[int] = []
        # 图节点的实体数据
        entity_list: list[str] = []
        # 时间戳索引列表
        timestamp_idx_list: list[int] = []
        # 位置列表
        position_list: list[Tuple[float, float, float, float]] = []
        # 与最新小图的距离
        distance_to_last_graph_list: list[int] = []
        # 中心时间戳的掩码
        center_node_mask = []
        # 最后一个时间戳的掩码
        last_node_mask = []

        # 获取基准时间戳的信息
        # 同一时间的实体顺序，保证前后连接边的时候是同一个实体
        anchor_entity_sequence_list: list[str] = []
        anchor_timestamp: Optional[str] = None
        for timestamp in time_context[
            # 仅从中间的小图中选，这样最可能让两边复制的节点数更少
            (self.graph_time_steps * (self.graph_time_num // 2)) : (
                self.graph_time_steps * (self.graph_time_num // 2 + 1)
            )
        ]:
            context_entities = self.store.get_speaker_context(
                video_id, entity_id, timestamp, None
            )
            # 选最长的实体上下文
            if len(context_entities) > len(anchor_entity_sequence_list):
                anchor_entity_sequence_list = context_entities
                anchor_timestamp = timestamp
        assert anchor_timestamp is not None
        anchor_video_data = self.store.get_video_data(
            video_id, entity_id, timestamp, None, self.cache
        )
        while len(anchor_entity_sequence_list) < self.max_context:
            # # 不够就补齐，空的
            # anchor_entity_sequence_list.append("")
            # anchor_video_data[0].append(np.zeros_like(anchor_video_data[0][0]))
            # anchor_video_data[1].append(
            #     np.zeros_like(anchor_video_data[1][0])
            #     if anchor_video_data[1][0] is not None
            #     else None
            # )
            # anchor_video_data[2].append(0)
            # anchor_video_data[3].append("")
            # anchor_video_data[4].append((0, 0, 0, 0))
            # 不够就补齐，只有自己复制自己，不然随机取
            entity = (
                random.choice(anchor_entity_sequence_list[1:])
                if len(anchor_entity_sequence_list) > 1
                else anchor_entity_sequence_list[0]
            )
            idx = anchor_entity_sequence_list.index(entity)
            anchor_entity_sequence_list.append(entity)
            anchor_video_data[0].append(anchor_video_data[0][idx].copy())
            anchor_video_data[1].append(
                anchor_video_data[1][idx].copy()  # type: ignore
                if anchor_video_data[1][idx] is not None
                else None
            )
            anchor_video_data[2].append(anchor_video_data[2][idx])
            anchor_video_data[3].append(entity)
            anchor_video_data[4].append(anchor_video_data[4][idx])

        # 对每个上下文时间戳，获取视频特征和标签
        for time_idx, timestamp in enumerate(time_context):
            if timestamp == anchor_timestamp:
                # 用基准时间戳的信息
                (
                    raw_video_data,
                    raw_video_vf_data,
                    raw_target_v,
                    raw_entities_v,
                    raw_positions,
                ) = anchor_video_data
            else:
                # 获取视频特征和标签
                (
                    raw_video_data,
                    raw_video_vf_data,
                    raw_target_v,
                    raw_entities_v,
                    raw_positions,
                ) = self.store.get_video_data(
                    video_id, entity_id, timestamp, self.max_context, self.cache
                )

            # 根据同一时间的实体顺序转换视频特征和标签
            video_data, video_vf_data, target_v, entities_v, positions = (
                [],
                [],
                [],
                [],
                [],
            )
            for entity in anchor_entity_sequence_list:
                if entity in raw_entities_v:
                    # 本时刻有就用本时刻的
                    idx = raw_entities_v.index(entity)
                    video_data.append(raw_video_data[idx])
                    video_vf_data.append(raw_video_vf_data[idx])
                    target_v.append(raw_target_v[idx])
                    entities_v.append(raw_entities_v[idx])
                    positions.append(raw_positions[idx])
                else:
                    # 本时刻没有
                    if entity in entity_list:
                        # 之前有就用之前的，为 None 是交给后面处理
                        video_data.append(None)
                        video_vf_data.append(None)
                        target_v.append(None)
                        entities_v.append(entity)
                        positions.append(None)
                    else:
                        # 之前没有，就用 anchor 的
                        idx = anchor_video_data[3].index(entity)
                        video_data.append(anchor_video_data[0][idx].copy())
                        video_vf_data.append(
                            anchor_video_data[1][idx].copy()  # type: ignore
                            if anchor_video_data[1][idx] is not None
                            else None
                        )
                        target_v.append(anchor_video_data[1][idx])
                        entities_v.append(anchor_video_data[2][idx])
                        positions.append(anchor_video_data[3][idx])
            # 限制上下文长度
            video_data, video_vf_data, target_v, entities_v, positions = (
                video_data[: self.max_context],
                video_vf_data[: self.max_context],
                target_v[: self.max_context],
                entities_v[: self.max_context],
                positions[: self.max_context],
            )

            # 获取音频特征和标签
            audio_data, audio_vf_data, target_a, entity_a = self.store.get_audio_data(
                video_id, entity_id, timestamp, self.cache
            )

            # 生成的 train feat 总共在磁盘里约 11G，valid feat 总共在磁盘里约 3G
            # 总共的帧数约为 100,0000，实体数约为 3,0000
            # 限制 self.entity_cache 长度，最大 40000(全部加载进内存)，从旧的开始删除
            while len(self.cache) > 40000:
                self.cache.pop(list(self.cache.keys())[0])

            a_feat = torch.from_numpy(audio_data)
            a_vf_feat = (
                torch.from_numpy(audio_vf_data) if audio_vf_data is not None else None
            )
            audio_feature_idx = len(feature_list)
            for i, (v_np, v_vf_np, target, entity, pos) in enumerate(
                zip(video_data, video_vf_data, target_v, entities_v, positions)
            ):
                if v_np is None:
                    # 对应：之前有就用之前的，为 None 是交给后面处理
                    idx = entity_list.index(entity)
                    v_np = feature_list[idx][1].numpy().copy()
                    v_vf_np = (
                        feature_list[idx][3].numpy().copy()
                        if feature_list[idx].size(0) > 2
                        else None
                    )
                    target = target_list[idx]
                    pos = position_list[idx]

                v_feat = torch.from_numpy(v_np)
                v_vf_feat = torch.from_numpy(v_vf_np) if v_vf_np is not None else None
                a_feat_dim = a_feat.size(0)
                v_feat_dim = v_feat.size(0)
                if a_vf_feat is not None and v_vf_feat is not None:
                    a_vf_feat_dim = a_vf_feat.size(0)
                    v_vf_feat_dim = v_vf_feat.size(0)
                    max_feat_dim = max(
                        a_feat_dim, v_feat_dim, a_vf_feat_dim, v_vf_feat_dim
                    )
                    # 维度不够的在后面填充 0
                    a_feat = F.pad(a_feat, (0, max_feat_dim - a_feat_dim))
                    v_feat = F.pad(v_feat, (0, max_feat_dim - v_feat_dim))
                    a_vf_feat = F.pad(a_vf_feat, (0, max_feat_dim - a_vf_feat_dim))
                    v_vf_feat = F.pad(v_vf_feat, (0, max_feat_dim - v_vf_feat_dim))
                    if i == 0:
                        # 第一个节点之前加一个纯音频节点
                        feature_list.append(
                            torch.stack(
                                [
                                    a_feat,
                                    torch.zeros_like(v_feat),
                                    a_vf_feat,
                                    torch.zeros_like(v_vf_feat),
                                ],
                                dim=0,
                            )
                        )
                    feature_list.append(
                        torch.stack([a_feat, v_feat, a_vf_feat, v_vf_feat], dim=0)
                    )
                else:
                    max_feat_dim = max(a_feat_dim, v_feat_dim)
                    a_feat = F.pad(a_feat, (0, max_feat_dim - a_feat_dim))
                    v_feat = F.pad(v_feat, (0, max_feat_dim - v_feat_dim))
                    if i == 0:
                        # 第一个节点之前加一个纯音频节点
                        feature_list.append(
                            torch.stack([a_feat, torch.zeros_like(v_feat)], dim=0)
                        )
                    feature_list.append(torch.stack([a_feat, v_feat], dim=0))
                if i == 0:
                    target_list.append(target_a)
                    audio_feature_mask.append(True)
                    audio_feature_idx_list.append(audio_feature_idx)
                    entity_list.append("")
                    timestamp_idx_list.append(time_idx)
                    position_list.append((0, 0, 1, 1))
                    distance_to_last_graph_list.append(
                        (self.graph_time_steps * self.graph_time_num - 1 - time_idx)
                        // self.graph_time_steps
                    )
                    center_node_mask.append(
                        time_idx == (len(time_context) - 1 - self.graph_time_steps // 2)
                    )
                    last_node_mask.append(time_idx == len(time_context) - 1)
                target_list.append(target)
                audio_feature_mask.append(False)
                audio_feature_idx_list.append(audio_feature_idx)
                entity_list.append(entity)
                timestamp_idx_list.append(time_idx)
                position_list.append(pos)
                distance_to_last_graph_list.append(
                    (self.graph_time_steps * self.graph_time_num - 1 - time_idx)
                    // self.graph_time_steps
                )
                center_node_mask.append(
                    time_idx == (len(time_context) - 1 - self.graph_time_steps // 2)
                )
                last_node_mask.append(time_idx == len(time_context) - 1)

        # 边的出发点，每一条无向边会正反记录两次
        source_vertices: list[int] = []
        # 边的结束点，每一条无向边会正反记录两次
        target_vertices: list[int] = []
        # 边出发点的位置信息，x1, y1, x2, y2
        source_vertices_pos: list[Tuple[float, float, float, float]] = []
        # 边结束点的位置信息，x1, y1, x2, y2
        target_vertices_pos: list[Tuple[float, float, float, float]] = []
        # 边出发点是否是音频特征
        source_vertices_audio: list[int] = []
        # 边结束点是否是音频特征
        target_vertices_audio: list[int] = []
        # 边的时间差比例
        time_delta_rate: list[float] = []
        # 边的时间差
        time_delta: list[int] = []
        # 是否自己连接边
        self_connect: list[int] = []

        # 构造边
        for i, (entity_i, timestamp_idx_i) in enumerate(
            zip(entity_list, timestamp_idx_list)
        ):
            for j, (entity_j, timestamp_idx_j) in enumerate(
                zip(entity_list, timestamp_idx_list)
            ):
                if timestamp_idx_i == timestamp_idx_j:
                    # 同一时刻上下文中的实体之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                    source_vertices_audio.append(1 if audio_feature_mask[i] else 0)
                    target_vertices_audio.append(1 if audio_feature_mask[j] else 0)
                    time_delta_rate.append(
                        abs(timestamp_idx_i - timestamp_idx_j) / self.graph_time_steps
                    )
                    time_delta.append(
                        abs(timestamp_idx_i - timestamp_idx_j) * self.graph_time_stride
                    )
                    self_connect.append(int(i == j))
                else:
                    # 超过了时间步数，不连接
                    if abs(timestamp_idx_i - timestamp_idx_j) > self.graph_time_steps:
                        continue

                    # 只单向连接
                    if not self.is_edge_double and timestamp_idx_i > timestamp_idx_j:
                        continue

                    # if entity_i == entity_j:
                    # 使用间隔判断是否同一实体，避免空白实体交叉连接
                    if abs(i - j) % (self.max_context + 1) == 0:
                        # 同一实体在不同时刻之间的连接
                        source_vertices.append(i)
                        target_vertices.append(j)
                        source_vertices_pos.append(position_list[i])
                        target_vertices_pos.append(position_list[j])
                        source_vertices_audio.append(1 if audio_feature_mask[i] else 0)
                        target_vertices_audio.append(1 if audio_feature_mask[j] else 0)
                        time_delta_rate.append(
                            abs(timestamp_idx_i - timestamp_idx_j)
                            / self.graph_time_steps
                        )
                        time_delta.append(
                            abs(timestamp_idx_i - timestamp_idx_j)
                            * self.graph_time_stride
                        )
                        self_connect.append(int(i == j))
                        if last_node_mask[j]:
                            # 如果是最后一个时间戳的节点，再连接一次
                            source_vertices.append(source_vertices[-1])
                            target_vertices.append(target_vertices[-1])
                            source_vertices_pos.append(source_vertices_pos[-1])
                            target_vertices_pos.append(target_vertices_pos[-1])
                            source_vertices_audio.append(source_vertices_audio[-1])
                            target_vertices_audio.append(target_vertices_audio[-1])
                            time_delta_rate.append(time_delta_rate[-1])
                            time_delta.append(time_delta[-1])
                            self_connect.append(self_connect[-1])
                    elif self.is_edge_across_entity:
                        # 不同实体在不同时刻之间的连接
                        source_vertices.append(i)
                        target_vertices.append(j)
                        source_vertices_pos.append(position_list[i])
                        target_vertices_pos.append(position_list[j])
                        source_vertices_audio.append(1 if audio_feature_mask[i] else 0)
                        target_vertices_audio.append(1 if audio_feature_mask[j] else 0)
                        time_delta_rate.append(
                            abs(timestamp_idx_i - timestamp_idx_j)
                            / self.graph_time_steps
                        )
                        time_delta.append(
                            abs(timestamp_idx_i - timestamp_idx_j)
                            * self.graph_time_stride
                        )
                        self_connect.append(int(i == j))

        entity_idx_list = [
            (self.store.entity_list.index((video_id, entity)) if entity != "" else -1)
            for entity in entity_list
        ]  # 视频实体序号，-1 代表纯音频节点没有视频实体

        return Data(
            # 维度为 [节点数量, 4 or 2, n]，表示每个节点的音频、视频特征、音频音脸嵌入、视频音脸嵌入
            x=torch.stack(feature_list, dim=0),
            # 维度为 [2, 边的数量]，表示每条边的两侧节点的索引
            edge_index=torch.tensor(
                [source_vertices, target_vertices], dtype=torch.long
            ),
            # 维度为 [边的数量, 6, 4]，表示每条边的两侧节点的位置信息、两侧节点是否纯音频节点、时间差比例、时间差、是否自己连接
            edge_attr=torch.tensor(
                [
                    source_vertices_pos,
                    target_vertices_pos,
                    [
                        (s_audio, t_audio, 0, 0)
                        for s_audio, t_audio in zip(
                            source_vertices_audio, target_vertices_audio
                        )
                    ],
                    [(rate, 0, 0, 0) for rate in time_delta_rate],
                    [(delta, 0, 0, 0) for delta in time_delta],
                    [(self_c, 0, 0, 0) for self_c in self_connect],
                ],
                dtype=torch.float,
            ).transpose(0, 1),
            # 维度为 [节点数量, 2]，每个节点的标签、实体标签
            y=torch.tensor(
                [
                    (target, entity)
                    for target, entity in zip(target_list, entity_idx_list)
                ]
            ),
            # 与最新小图的距离
            distance_to_last_graph_list=distance_to_last_graph_list,
            # 中间时间戳的掩码
            center_node_mask=center_node_mask,
            # 最后一个时间戳的掩码
            last_node_mask=last_node_mask,
            # 纯音频节点的掩码
            audio_node_mask=audio_feature_mask,
            # 节点的时刻对应纯音频节点的索引
            audio_feature_idx_list=audio_feature_idx_list,
        )
