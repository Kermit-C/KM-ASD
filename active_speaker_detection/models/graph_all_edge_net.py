#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.parameter
from torch_geometric.nn import BatchNorm, EdgeConv
from torch_geometric.utils import dropout_adj


class LinearPathPreact(nn.Module):
    """EdgeConv 的线性路径预激活模块"""

    def __init__(self, in_channels, hidden_channels):
        super(LinearPathPreact, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x1 = self.fc1(x1)

        x2 = self.bn2(x1)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)

        return x2


class GraphAllEdgeNet(nn.Module):

    def __init__(self, encoder, channels):
        super().__init__()

        self.encoder = encoder
        self.av_fusion = nn.Linear(128 * 2, 128)

        self.layer_1 = EdgeConv(LinearPathPreact(128 * 2, channels))
        self.batch_1 = BatchNorm(channels)
        self.layer_2 = EdgeConv(LinearPathPreact(channels * 2, channels))
        self.batch_2 = BatchNorm(channels)
        self.layer_3 = EdgeConv(LinearPathPreact(channels * 2, channels))
        self.batch_3 = BatchNorm(channels)
        self.layer_4 = EdgeConv(LinearPathPreact(channels * 2, channels))

        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dropout_edge = 0.2

    def forward(
        self,
        data,
        audio_size: Optional[Tuple[int, int, int]] = None,
    ):
        x, edge_index, edge_attr, _ = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if len(x.shape) > 3:
            # 从数据中提取音频和视频特征
            assert audio_size is not None
            audio_data = (
                x[:, 0, 0, 0, : audio_size[1], : audio_size[2]]
                .unsqueeze(1)
                .unsqueeze(1)
            )
            video_data = x[:, 1, :, :, :, :].unsqueeze(1)
            audio_feats, video_feats, audio_out, video_out, _, vf_a_emb, vf_v_emb = (
                self.encoder(audio_data, video_data)
            )

            # 图特征
            graph_feats = self.av_fusion(torch.cat([audio_feats, video_feats], dim=1))
        else:
            # 输入的就是 encoder 出来的 128 维特征
            audio_out = None
            video_out = None
            vf_a_emb = None
            vf_v_emb = None
            graph_feats = self.av_fusion(
                torch.cat([x[:, 0, :].unsqueeze(1), x[:, 1, :].unsqueeze(1)], dim=1)
            )

        edge_index_1, edge_attr_1 = dropout_adj(
            edge_index=edge_index,
            edge_attr=edge_attr,
            p=self.dropout_edge,
            training=self.training,
        )
        graph_feats_1 = self.layer_1(graph_feats, edge_index)
        graph_feats_1 = self.batch_1(graph_feats_1)
        graph_feats_1 = self.relu(graph_feats_1)
        graph_feats_1 = self.dropout(graph_feats_1)

        graph_feats_2 = self.layer_2(graph_feats_1, edge_index)
        graph_feats_2 += graph_feats_1
        graph_feats_2 = self.batch_2(graph_feats_2)
        graph_feats_2 = self.relu(graph_feats_2)
        graph_feats_2 = self.dropout(graph_feats_2)

        graph_feats_3 = self.layer_3(graph_feats_2, edge_index)
        graph_feats_3 += graph_feats_2
        graph_feats_3 = self.batch_3(graph_feats_3)
        graph_feats_3 = self.relu(graph_feats_3)
        graph_feats_3 = self.dropout(graph_feats_3)

        graph_feats_4 = self.layer_4(graph_feats_3, edge_index)
        graph_feats_4 += graph_feats_3

        out = self.fc(graph_feats_4)

        return out, audio_out, video_out, vf_a_emb, vf_v_emb
