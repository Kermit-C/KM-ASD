#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter
from torch_geometric.nn import BatchNorm, EdgeConv, GATConv, GATv2Conv
from torch_geometric.utils import dropout_adj

from active_speaker_detection.utils.vf_util import cosine_similarity

from ..utils.spatial_grayscale_util import batch_create_spatial_grayscale
from .graph_all_edge_net import LinearPathPreact


class GraphGatEdgeNet(nn.Module):

    def __init__(
        self,
        in_a_channels,
        in_v_channels,
        in_vf_channels,
        channels,
        edge_attr_dim,
        is_gatv2: bool = False,
    ):
        super().__init__()

        self.in_a_channels = in_a_channels
        self.in_v_channels = in_v_channels
        self.in_vf_channels = in_vf_channels

        self.av_fusion = nn.Linear(in_a_channels + in_v_channels, channels)
        self.batch_0 = BatchNorm(channels)

        if not is_gatv2:
            self.layer_1 = GATConv(
                channels,
                channels,
                heads=8,
                edge_dim=edge_attr_dim,
                dropout=0.2,
                concat=True,
                negative_slope=0.2,
                bias=True,
            )
        else:
            self.layer_1 = GATv2Conv(
                channels,
                channels,
                heads=8,
                edge_dim=edge_attr_dim,
                dropout=0.2,
                concat=True,
                negative_slope=0.2,
                bias=True,
            )
        self.batch_1 = BatchNorm(channels * 8)
        self.layer_2 = EdgeConv(
            LinearPathPreact(channels * 8 * 2, channels), aggr="mean"
        )
        self.batch_2 = BatchNorm(channels)
        self.layer_3 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")
        self.batch_3 = BatchNorm(channels)
        self.layer_4 = EdgeConv(LinearPathPreact(channels * 2, channels), aggr="mean")

        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dropout_edge = 0.2

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

        return out
