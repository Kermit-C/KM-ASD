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
from torchvision import transforms

from active_speaker_detection.utils.augmentations import video_temporal_crop

from .data_store import DataStore


class End2endDataset(Dataset):

    def __init__(
        self,
        audio_root,
        video_root,
        data_store_train_cache,
        clip_length,  # 片段长度，短时序上下文片段的长度
        graph_time_steps: int,  # 图的时间步数
        graph_time_stride: int,  # 步长
        is_edge_double=False,  # 是否边是双向的
        is_edge_across_entity=False,  # 是否跨实体连接，即不同实体不同时刻之间的连接
        max_context: Optional[int] = None,  # 最大上下文大小，即上下文中有多少个实体
        video_transform: Optional[transforms.Compose] = None,  # 视频转换方法
        do_video_augment=False,  # 是否视频增强
        crop_ratio=0.95,  # 视频裁剪比例
        norm_audio=False,  # 是否归一化音频
    ):
        super().__init__()
        self.store = DataStore(audio_root, video_root, data_store_train_cache)

        # 后处理
        self.crop_ratio = crop_ratio
        self.video_transform = (
            video_transform if video_transform is not None else transforms.ToTensor()
        )
        self.do_video_augment = do_video_augment  # 是否视频增强

        # 图节点的配置
        self.norm_audio = norm_audio  # 是否归一化音频
        self.half_clip_length = math.floor(
            clip_length / 2
        )  # 每刻计算特征的帧数一半的长度

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
        # 获取时间上下文，时间戳列表
        time_context: List[str] = self.store.get_time_context(
            target_entity_metadata,
            center_index,
            self.graph_time_steps,  # 和 dataset_graph.py 不同，这里如果形成大图的话，内存不够
            self.graph_time_stride,
        )

        # 图节点特征
        feature_list: list[torch.Tensor] = []
        # 图节点的标签数据，前面是音频标签，后面是视频标签
        target_list: list[Tuple[int, int]] = []
        # 图节点的实体数据
        entity_list: list[str] = []
        # 图节点的音频实体数据
        audio_entity_list: list[str] = []
        # 时间戳列表
        timestamp_list: list[str] = []
        # 位置列表
        position_list: list[Tuple[float, float, float, float]] = []
        # 最后一个时间戳的掩码
        center_node_mask = []

        # 所有时间戳
        all_ts = [ted[1] for ted in target_entity_metadata]
        cache = {}  # 以图片路径为 key 的缓存
        # 对每个上下文时间戳，获取视频特征和标签
        for time_idx, timestamp in enumerate(time_context):
            target_index = all_ts.index(timestamp)
            # 获取视频特征和标签
            video_data, target_v, entities_v, positions = self._get_scene_video_data(
                video_id, entity_id, target_index, cache
            )
            # 获取音频特征和标签
            audio_data, audio_fbank, target_a, entity_a = self.store.get_audio_data(
                video_id,
                entity_id,
                target_index,
                self.half_clip_length * 2,  # 音频长度取两倍
            )

            a_data = torch.from_numpy(audio_data)
            for v_data, target, entity, pos in zip(
                video_data, target_v, entities_v, positions
            ):
                # 一个数据是 2 * 通道数 * clip数 * 高度 * 宽度 的 5D tensor
                # 前面是音频特征，后面是视频特征
                # 音频特征通道数和 clip 数都是 1
                a_extend_data = torch.zeros_like(v_data)
                a_extend_data[0, 0, : a_data.size(1), : a_data.size(2)] = a_data
                feature_list.append(torch.stack([a_extend_data, v_data], dim=0))
                target_list.append((target_a, target))
                entity_list.append(entity)
                audio_entity_list.append(entity_a)
                timestamp_list.append(timestamp)
                position_list.append(pos)
                center_node_mask.append(time_idx == (len(time_context) - 1) // 2)

        # 边的出发点，每一条无向边会正反记录两次
        source_vertices: list[int] = []
        # 边的结束点，每一条无向边会正反记录两次
        target_vertices: list[int] = []
        # 边出发点的位置信息，x1, y1, x2, y2
        source_vertices_pos: list[Tuple[float, float, float, float]] = []
        # 边结束点的位置信息，x1, y1, x2, y2
        target_vertices_pos: list[Tuple[float, float, float, float]] = []
        # 边的时间差比例
        time_delta_rate: list[float] = []

        # 构造边
        for i, (entity, timestamp) in enumerate(zip(entity_list, timestamp_list)):
            for j, (entity, timestamp) in enumerate(zip(entity_list, timestamp_list)):
                # # 自己不连接自己
                # if i == j:
                #     continue

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
                    time_delta_rate.append(abs(i - j) / self.graph_time_steps)
                elif entity_list[i] == entity_list[j]:
                    # 同一实体在不同时刻之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                    time_delta_rate.append(abs(i - j) / self.graph_time_steps)
                elif self.is_edge_across_entity:
                    # 不同实体在不同时刻之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                    time_delta_rate.append(abs(i - j) / self.graph_time_steps)

        entity_idx_list = [
            self.store.entity_list.index((video_id, entity)) for entity in entity_list
        ]
        audio_entity_idx_list = [
            (self.store.entity_list.index((video_id, entity)) if entity != "" else -1)
            for entity in audio_entity_list
        ]  # 音频实体序号，-1 代表没有，则是环境音

        return Data(
            # 维度为 [节点数量, 2, 通道数 , clip数 , 高度 , 宽度]，表示每个节点的音频和视频特征
            x=torch.stack(feature_list, dim=0),
            # 维度为 [2, 边的数量]，表示每条边的两侧节点的索引
            edge_index=torch.tensor(
                [source_vertices, target_vertices], dtype=torch.long
            ),
            # 维度为 [边的数量, 3, 4]，表示每条边的两侧节点的位置信息，最后一个维度是时间差比例
            edge_attr=torch.tensor(
                [
                    source_vertices_pos,
                    target_vertices_pos,
                    [(rate, 0, 0, 0) for rate in time_delta_rate],
                ],
                dtype=torch.float,
            ).transpose(0, 1),
            # 维度为 [节点数量, 4]，表示每个节点的标签，前面是音频标签，中间是视频标签，后面是音频实体标签和视频实体标签
            y=torch.tensor(
                [
                    (target[0], target[1], entity_a, entity)
                    for target, entity, entity_a in zip(
                        target_list,
                        audio_entity_idx_list,
                        entity_idx_list,
                    )
                ]
            ),
            # 最后一个时间戳的掩码
            center_node_mask=center_node_mask,
        )

    def get_audio_size(
        self,
    ) -> Tuple[Tuple[int, int, int]]:
        """获得音频的大小，返回一个元组(1, 13, T)"""
        return self.store.get_audio_size(self.half_clip_length)

    def _get_scene_video_data(
        self, video_id: str, entity_id: str, mid_index: int, cache: dict
    ) -> Tuple[
        list[torch.Tensor],
        list[int],
        list[str],
        list[Tuple[float, float, float, float]],
    ]:
        """根据实体 ID 和中间某刻时间戳索引
        获取所有人视频画面 list(人数) tensor(通道数 * clip数 * 高度 * 宽度) 和这一刻所有实体的标签
        :param video_id: 视频 id
        :param entity_id: 实体 id
        :param mid_index: 时间戳索引，即中间时间戳的索引
        :param cache: 缓存
        """
        video_data, targets, entities, positions = self.store.get_video_data(
            video_id,
            entity_id,
            mid_index,
            self.max_context,
            self.half_clip_length,
            self.video_transform,
            video_temporal_crop if self.do_video_augment else None,
            self.crop_ratio if self.do_video_augment else None,
            cache,
        )
        # 把一个人的多个画面合并成一个 4D tensor
        return (
            [torch.stack(vd, dim=1) for vd in video_data],
            targets,
            entities,
            positions,
        )
