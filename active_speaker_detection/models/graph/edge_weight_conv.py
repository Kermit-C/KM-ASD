#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-29 19:03:21
"""

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor


class EdgeWeightConv(MessagePassing):

    def __init__(
        self, nn: Callable, node_dim: int, edge_dim: int, aggr: str = "mean", **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.lin_edge = torch.nn.Linear(edge_dim, node_dim)
        self.fusion_edge = torch.nn.Linear(node_dim, node_dim)

        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(node_dim)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.lin_edge.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor) -> Tensor:  # type: ignore
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        edge_attr = self.relu(self.lin_edge(edge_attr))

        x_i_edge = x_i + edge_attr
        x_i_edge = self.fusion_edge(x_i_edge)
        x_i_edge = self.relu(self.bn(x_i_edge))
        x_j_edge = x_j + edge_attr
        x_j_edge = self.fusion_edge(x_j_edge)
        x_j_edge = self.relu(self.bn(x_j_edge))

        return self.nn(torch.cat([x_i, x_j_edge - x_i_edge], dim=-1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"
