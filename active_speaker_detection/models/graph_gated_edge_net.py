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
import torch.nn.parameter
from torch_geometric.nn import BatchNorm, EdgeConv, GatedGraphConv
from torch_geometric.utils import dropout_adj

from .graph_all_edge_net import LinearPathPreact
from .spatial_extract.spatial_grayscale_util import batch_create_spatial_grayscale


class GraphGatedEdgeNet(nn.Module):

    def __init__(
        self,
        channels,
        spatial_net: Optional[nn.Module] = None,  # 为空则不使用空间关系网络
        spatial_feature_dim: int = 64,
        spatial_grayscale_width: int = 112,
        spatial_grayscale_height: int = 112,
    ):
        super().__init__()

        self.spatial_net = spatial_net
        self.spatial_feature_dim = spatial_feature_dim
        self.spatial_grayscale_width = spatial_grayscale_width
        self.spatial_grayscale_height = spatial_grayscale_height

        self.av_fusion = nn.Linear(128 * 2, 128)
        self.edge_weight_fc = nn.Linear(spatial_feature_dim, 1)

        self.layer_1 = GatedGraphConv(channels, num_layers=2, aggr="mean", bias=True)
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
        self.dropout_edge = 0.2

    def forward(self, data):
        x, edge_index, edge_attr_info, _ = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        if self.spatial_net is not None:
            spatial_grayscale_imgs = batch_create_spatial_grayscale(
                edge_attr_info[:, :2],
                self.spatial_grayscale_width,
                self.spatial_grayscale_height,
                3,
            )
            edge_attr = self.spatial_net(spatial_grayscale_imgs)
        else:
            # 权重全为 0，代表没有空间关系
            edge_attr = torch.zeros(
                edge_index.size(1), self.spatial_feature_dim, device=x.device
            )
        edge_attr = self.edge_weight_fc(edge_attr)

        graph_feats = self.av_fusion(
            torch.cat([x[:, 0, :].unsqueeze(1), x[:, 1, :].unsqueeze(1)], dim=1)
        )

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
