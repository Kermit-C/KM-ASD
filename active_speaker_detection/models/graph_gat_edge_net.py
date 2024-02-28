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
from torch_geometric.nn import BatchNorm, EdgeConv, GATConv, GATv2Conv
from torch_geometric.utils import dropout_adj

from .graph_all_edge_net import LinearPathPreact
from .spatial_extract.spatial_grayscale_util import batch_create_spatial_grayscale


class GraphGatEdgeNet(nn.Module):

    def __init__(
        self,
        channels,
        spatial_net: Optional[nn.Module] = None,  # 为空则不使用空间关系网络
        spatial_feature_dim: int = 64,
        spatial_grayscale_width: int = 112,
        spatial_grayscale_height: int = 112,
        is_gatv2: bool = False,
    ):
        super().__init__()

        self.spatial_net = spatial_net
        self.spatial_feature_dim = spatial_feature_dim
        self.spatial_grayscale_width = spatial_grayscale_width
        self.spatial_grayscale_height = spatial_grayscale_height
        self.spatial_mini_batch = 32  # 空间网络的 mini batch 大小，因为图节点数太大了（上千），所以需要分批次计算

        self.av_fusion = nn.Linear(128 * 2, 128)

        if not is_gatv2:
            self.layer_1 = GATConv(
                channels,
                channels,
                heads=8,
                edge_dim=spatial_feature_dim,
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
                edge_dim=spatial_feature_dim,
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
            spatial_feats = []
            for i in range(0, spatial_grayscale_imgs.size(0), self.spatial_mini_batch):
                spatial_feats.append(
                    self.spatial_net(
                        spatial_grayscale_imgs[i : i + self.spatial_mini_batch]
                    )
                )
            edge_attr = torch.cat(spatial_feats, dim=0)
        else:
            # 权重全为 0，代表没有空间关系
            edge_attr = torch.zeros(
                edge_index.size(1), self.spatial_feature_dim, device=x.device
            )

        graph_feats = self.av_fusion(
            torch.cat([x[:, 0, :].squeeze(1), x[:, 1, :].squeeze(1)], dim=1)
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
