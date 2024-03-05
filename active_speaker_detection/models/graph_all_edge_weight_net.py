#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter
from torch_geometric.nn import BatchNorm
from torch_geometric.utils import dropout_adj

from active_speaker_detection.models.graph.edge_weight_conv import EdgeWeightConv
from active_speaker_detection.utils.vf_util import cosine_similarity


class LinearPathPreact(nn.Module):
    """EdgeWeightConv 的线性路径预激活模块"""

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


class GraphAllEdgeWeightNet(nn.Module):

    def __init__(
        self,
        in_a_channels,
        in_v_channels,
        in_vf_channels,
        channels,
        edge_attr_dim,
    ):
        super().__init__()

        self.in_a_channels = in_a_channels
        self.in_v_channels = in_v_channels
        self.in_vf_channels = in_vf_channels

        self.av_fusion = nn.Linear(in_a_channels + in_v_channels, channels)
        self.batch_0 = BatchNorm(channels)

        self.layer_1 = EdgeWeightConv(
            LinearPathPreact(channels * 2, channels),
            node_dim=channels,
            edge_dim=edge_attr_dim,
            aggr="mean",
        )
        self.batch_1 = BatchNorm(channels)
        self.layer_2 = EdgeWeightConv(
            LinearPathPreact(channels * 2, channels),
            node_dim=channels,
            edge_dim=edge_attr_dim,
            aggr="mean",
        )
        self.batch_2 = BatchNorm(channels)
        self.layer_3 = EdgeWeightConv(
            LinearPathPreact(channels * 2, channels),
            node_dim=channels,
            edge_dim=edge_attr_dim,
            aggr="mean",
        )
        self.batch_3 = BatchNorm(channels)
        self.layer_4 = EdgeWeightConv(
            LinearPathPreact(channels * 2, channels),
            node_dim=channels,
            edge_dim=edge_attr_dim,
            aggr="mean",
        )

        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dropout_edge = 0

    def forward(self, data):
        x, edge_index, edge_attr, _ = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        audio_feats = x[:, 0, : self.in_a_channels].squeeze(1)
        video_feats = x[:, 1, : self.in_v_channels].squeeze(1)
        audio_vf_emb = (
            x[:, 2, : self.in_vf_channels].squeeze(1) if x.size(1) > 2 else None
        )
        video_vf_emb = (
            x[:, 3, : self.in_vf_channels].squeeze(1) if x.size(1) > 3 else None
        )
        if audio_vf_emb is not None and video_vf_emb is not None:
            # sim 的维度是 (B, )
            sim = cosine_similarity(audio_vf_emb, video_vf_emb)
            audio_feats = audio_feats * sim.unsqueeze(1)
        graph_feats = self.av_fusion(torch.cat([audio_feats, video_feats], dim=1))
        graph_feats = self.batch_0(graph_feats)
        graph_feats = self.relu(graph_feats)

        edge_index_1, edge_attr_1 = dropout_adj(
            edge_index=edge_index,
            edge_attr=edge_attr,
            p=self.dropout_edge,
            training=self.training,
        )
        graph_feats_1 = self.layer_1(graph_feats, edge_index_1, edge_attr_1)
        graph_feats_1 = self.batch_1(graph_feats_1)
        graph_feats_1 = self.relu(graph_feats_1)
        graph_feats_1 = self.dropout(graph_feats_1)

        graph_feats_2 = self.layer_2(graph_feats_1, edge_index, edge_attr)
        graph_feats_2 += graph_feats_1
        graph_feats_2 = self.batch_2(graph_feats_2)
        graph_feats_2 = self.relu(graph_feats_2)
        graph_feats_2 = self.dropout(graph_feats_2)

        graph_feats_3 = self.layer_3(graph_feats_2, edge_index, edge_attr)
        graph_feats_3 += graph_feats_2
        graph_feats_3 = self.batch_3(graph_feats_3)
        graph_feats_3 = self.relu(graph_feats_3)
        graph_feats_3 = self.dropout(graph_feats_3)

        graph_feats_4 = self.layer_4(graph_feats_3, edge_index, edge_attr)
        graph_feats_4 += graph_feats_3

        out = self.fc(graph_feats_4)

        return out
