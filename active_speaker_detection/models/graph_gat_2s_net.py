#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import torch
import torch.nn as nn
import torch.nn.parameter
from torch_geometric.nn import BatchNorm, GATConv, GATv2Conv


class GraphGatTwoStreamNet(nn.Module):

    def __init__(
        self,
        in_a_channels,
        in_v_channels,
        in_vf_channels,
        channels,
        is_gatv2: bool = False,
    ):
        super().__init__()

        self.in_a_channels = in_a_channels
        self.in_v_channels = in_v_channels
        self.in_vf_channels = in_vf_channels
        self.channels = channels

        self.layer_0_a = nn.Linear(in_a_channels, channels)
        self.layer_0_v = nn.Linear(in_v_channels, channels)
        self.batch_0_a = BatchNorm(channels)
        self.batch_0_v = BatchNorm(channels)

        if not is_gatv2:
            TargetGATConv = GATConv
        else:
            TargetGATConv = GATv2Conv

        self.layer_1_1 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.layer_1_2_1 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.layer_1_2_2 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.batch_1 = BatchNorm(channels)
        self.layer_2_1 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.layer_2_2_1 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.layer_2_2_2 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.batch_2 = BatchNorm(channels)
        self.layer_3_1 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.layer_3_2_1 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )
        self.layer_3_2_2 = TargetGATConv(
            channels,
            channels,
            heads=4,
            dropout=0.2,
            concat=False,
            negative_slope=0.2,
            bias=True,
            add_self_loops=False,
        )

        # 分类器
        self.fc_a = nn.Linear(channels, 2)
        self.fc_v = nn.Linear(channels, 2)
        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dropout_edge = 0

    def forward(self, data):
        x, edge_index, edge_delta, edge_self = (
            data.x,
            data.edge_index,
            data.edge_delta,
            data.edge_self,
        )
        audio_node_mask = []
        if type(data.audio_node_mask[0]) == bool:
            audio_node_mask = data.audio_node_mask
        else:
            for mask in data.audio_node_mask:
                audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        graph_feats = torch.zeros(x.size(0), self.channels, dtype=x.dtype).to(x.device)
        audio_feats = x[audio_node_mask][:, 0, : self.in_a_channels]
        video_feats = x[video_node_mask][:, 1, : self.in_v_channels]
        graph_feats[audio_node_mask] = self.batch_0_a(self.layer_0_a(audio_feats))
        graph_feats[video_node_mask] = self.batch_0_a(self.layer_0_v(video_feats))
        graph_feats = self.relu(graph_feats)

        distance1_mask = edge_delta < 1
        distance2_mask = ((edge_delta >= 1) & (edge_delta < 11)) | (edge_self == 1)
        distance3_mask = edge_delta >= 11 | (edge_self == 1)

        graph_feats_1_1 = graph_feats
        graph_feats_1_1 = self.layer_1_1(graph_feats_1_1, edge_index[:, distance1_mask])
        graph_feats_1_2 = graph_feats
        graph_feats_1_2 = self.layer_1_2_1(
            graph_feats_1_2, edge_index[:, distance2_mask]
        )
        graph_feats_1_2 = self.layer_1_2_2(
            graph_feats_1_2, edge_index[:, distance3_mask]
        )
        graph_feats_1 = graph_feats_1_1 + graph_feats_1_2 + graph_feats
        graph_feats_1 = self.batch_1(graph_feats_1)
        graph_feats_1 = self.relu(graph_feats_1)
        graph_feats_1 = self.dropout(graph_feats_1)

        graph_feats_2_1 = graph_feats_1
        graph_feats_2_1 = self.layer_2_1(graph_feats_2_1, edge_index[:, distance1_mask])
        graph_feats_2_2 = graph_feats_1
        graph_feats_2_2 = self.layer_2_2_1(
            graph_feats_2_2, edge_index[:, distance2_mask]
        )
        graph_feats_2_2 = self.layer_2_2_2(
            graph_feats_2_2, edge_index[:, distance3_mask]
        )
        graph_feats_2 = graph_feats_2_1 + graph_feats_2_2 + graph_feats_1
        graph_feats_2 = self.batch_2(graph_feats_2)
        graph_feats_2 = self.relu(graph_feats_2)
        graph_feats_2 = self.dropout(graph_feats_2)

        graph_feats_3_1 = graph_feats_2
        graph_feats_3_1 = self.layer_3_1(graph_feats_3_1, edge_index[:, distance1_mask])
        graph_feats_3_2 = graph_feats_2
        graph_feats_3_2 = self.layer_3_2_1(
            graph_feats_3_2, edge_index[:, distance2_mask]
        )
        graph_feats_3_2 = self.layer_3_2_2(
            graph_feats_3_2, edge_index[:, distance3_mask]
        )
        graph_feats_3 = graph_feats_3_1 + graph_feats_3_2 + graph_feats_2

        out = self.fc(graph_feats_3)
        audio_out = self.fc_a(graph_feats[audio_node_mask])
        video_out = self.fc_v(graph_feats[video_node_mask])

        return out, audio_out, video_out
