#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-24 12:38:39
"""

import torch
import torch.nn as nn


class TemporalShift(nn.Module):

    def __init__(self, net, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.fold_div = n_div

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.size()
        x = self.shift(x, fold_div=self.fold_div)
        x = x.view(b * t, c, h, w)
        x = self.net(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3))
        return x

    @staticmethod
    def shift(x, fold_div):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.size()

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # not shift

        return out.view(b, t, c, h, w)
        # out: (B, T, C, H, W)
