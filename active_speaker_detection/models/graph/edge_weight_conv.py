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
        self.lin_edge = nn.Linear(
            edge_dim, node_dim, bias=False, weight_initializer="glorot"
        )
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
        edge_attr = self.lin_edge(edge_attr)
        return self.nn(torch.cat([x_i, edge_attr + x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"
