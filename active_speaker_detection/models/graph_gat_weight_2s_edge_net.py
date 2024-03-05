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

from ..utils.vf_util import cosine_similarity
from .graph.gat_weight_conv import GatWeightConv
from .graph_all_edge_net import LinearPathPreact


class GraphGatWeightTwoStreamEdgeNet(nn.Module):

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
        self.channels = channels

        self.layer_0_a = nn.Linear(in_a_channels, channels)
        self.layer_0_v = nn.Linear(in_v_channels, channels)
        self.batch_0 = BatchNorm(channels)

        self.layer_1_1 = GatWeightConv(
            channels,
            channels,
            LinearPathPreact(channels * 2, channels),
            heads=4,
            edge_dim=edge_attr_dim,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
        )
        self.layer_1_2 = GatWeightConv(
            channels,
            channels,
            LinearPathPreact(channels * 2, channels),
            heads=4,
            edge_dim=edge_attr_dim,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
        )
        self.batch_1 = BatchNorm(channels)
        self.layer_2 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")
        self.batch_2 = BatchNorm(channels)
        self.layer_3 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")

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
        distance2_mask = ((edge_delta >= 1) & (edge_delta < 4)) | (edge_self == 1)
        distance3_mask = ((edge_delta >= 4) & (edge_delta < 8)) | (edge_self == 1)
        distance4_mask = ((edge_delta >= 8) & (edge_delta < 15)) | (edge_self == 1)
        distance5_mask = edge_delta >= 15 | (edge_self == 1)
        distance_mask_list = [
            distance1_mask,
            distance2_mask,
            distance3_mask,
            distance4_mask,
            distance5_mask,
        ]

        graph_feats_1_1 = graph_feats
        for distance_mask in distance_mask_list[:2]:
            graph_feats_1_1 = self.layer_1_1(
                graph_feats_1_1, edge_index[:, distance_mask], edge_attr[distance_mask]
            )
        graph_feats_1_2 = graph_feats
        for distance_mask in distance_mask_list[2:]:
            graph_feats_1_2 = self.layer_1_2(
                graph_feats_1_2, edge_index[:, distance_mask], edge_attr[distance_mask]
            )
        graph_feats_1 = graph_feats_1_1 + graph_feats_1_2 + graph_feats
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

        out = self.fc(graph_feats_3)

        return out
