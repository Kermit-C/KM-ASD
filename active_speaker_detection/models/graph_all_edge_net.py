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
from torch_geometric.nn import BatchNorm, EdgeConv
from torch_geometric.utils import dropout_adj

from active_speaker_detection.utils.vf_util import cosine_similarity


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

    def __init__(
        self,
        in_a_channels,
        in_v_channels,
        in_vf_channels,
        channels,
    ):
        super().__init__()

        self.in_a_channels = in_a_channels
        self.in_v_channels = in_v_channels
        self.in_vf_channels = in_vf_channels
        self.channels = channels

        self.layer_0_a = nn.Linear(in_a_channels, channels)
        self.layer_0_v = nn.Linear(in_v_channels, channels)
        # self.av_fusion = nn.Linear(channels * 2, channels)
        self.batch_0 = BatchNorm(channels)

        self.layer_1 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")
        self.batch_1 = BatchNorm(channels)
        self.layer_2 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")
        self.batch_2 = BatchNorm(channels)
        self.layer_3 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")
        self.batch_3 = BatchNorm(channels)
        self.layer_4 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")

        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dropout_edge = 0

    def forward(self, data):
        x, edge_index, edge_attr, edge_delta, edge_self = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.edge_delta,
            data.edge_self,
        )
        audio_node_mask = []
        for mask in data.audio_node_mask:
            audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        # audio_feats = x[:, 0, : self.in_a_channels].squeeze(1)
        # video_feats = x[:, 1, : self.in_v_channels].squeeze(1)
        # audio_vf_emb = (
        #     x[:, 2, : self.in_vf_channels].squeeze(1) if x.size(1) > 2 else None
        # )
        # video_vf_emb = (
        #     x[:, 3, : self.in_vf_channels].squeeze(1) if x.size(1) > 3 else None
        # )
        # if audio_vf_emb is not None and video_vf_emb is not None:
        #     # sim 的维度是 (B, )
        #     sim = cosine_similarity(audio_vf_emb, video_vf_emb)
        #     audio_feats = audio_feats * sim.unsqueeze(1)
        # audio_feats = self.layer_0_a(audio_feats)
        # video_feats = self.layer_0_v(video_feats)
        # graph_feats = self.av_fusion(torch.cat([audio_feats, video_feats], dim=1))
        # graph_feats = self.batch_0(graph_feats)
        # graph_feats = self.relu(graph_feats)

        graph_feats = torch.zeros(x.size(0), self.channels, dtype=x.dtype).to(x.device)
        audio_feats = x[:, 0, : self.in_a_channels][audio_node_mask]
        video_feats = x[:, 1, : self.in_v_channels][video_node_mask]
        graph_feats[audio_node_mask] = self.layer_0_a(audio_feats)
        graph_feats[video_node_mask] = self.layer_0_v(video_feats)
        graph_feats = self.batch_0(graph_feats)
        graph_feats = self.relu(graph_feats)

        graph_vf_emb = torch.zeros(x.size(0), self.in_vf_channels, dtype=x.dtype).to(
            x.device
        )
        audio_vf_emb = (
            x[:, 2, : self.in_vf_channels][audio_node_mask] if x.size(1) > 2 else None
        )
        video_vf_emb = (
            x[:, 3, : self.in_vf_channels][video_node_mask] if x.size(1) > 3 else None
        )
        if audio_vf_emb is not None and video_vf_emb is not None:
            graph_vf_emb[audio_node_mask] = audio_vf_emb
            graph_vf_emb[video_node_mask] = video_vf_emb

        distance1_mask = edge_delta < 1
        distance2_mask = ((edge_delta >= 1) & (edge_delta < 3)) | (edge_self == 1)
        distance3_mask = ((edge_delta >= 3) & (edge_delta < 8)) | (edge_self == 1)
        distance4_mask = ((edge_delta >= 8) & (edge_delta < 15)) | (edge_self == 1)
        distance5_mask = edge_delta >= 15 | (edge_self == 1)
        distance_mask_list = [
            distance1_mask,
            distance2_mask,
            distance3_mask,
            distance4_mask,
            distance5_mask,
        ]

        graph_feats_1 = graph_feats
        for distance_mask in distance_mask_list[:2]:
            edge_index_1, _ = dropout_adj(
                edge_index=edge_index[:, distance_mask],
                p=self.dropout_edge,
                training=self.training,
            )
            graph_feats_1 = self.layer_1(graph_feats_1, edge_index_1)
        graph_feats_1 = self.batch_1(graph_feats_1)
        graph_feats_1 = self.relu(graph_feats_1)
        graph_feats_1 = self.dropout(graph_feats_1)

        graph_feats_2 = graph_feats_1
        for distance_mask in distance_mask_list[:2]:
            graph_feats_2 = self.layer_2(graph_feats_2, edge_index[:, distance_mask])
        graph_feats_2 += graph_feats_1
        graph_feats_2 = self.batch_2(graph_feats_2)
        graph_feats_2 = self.relu(graph_feats_2)
        graph_feats_2 = self.dropout(graph_feats_2)

        graph_feats_3 = graph_feats_2
        for distance_mask in distance_mask_list[:2]:
            graph_feats_3 = self.layer_3(graph_feats_3, edge_index[:, distance_mask])
        graph_feats_3 += graph_feats_2
        graph_feats_3 = self.batch_3(graph_feats_3)
        graph_feats_3 = self.relu(graph_feats_3)
        graph_feats_3 = self.dropout(graph_feats_3)

        graph_feats_4 = graph_feats_3
        for distance_mask in distance_mask_list[:2]:
            graph_feats_4 = self.layer_4(graph_feats_4, edge_index[:, distance_mask])
        graph_feats_4 += graph_feats_3

        out = self.fc(graph_feats_4)

        return out
