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

    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div

    def forward(self, x, clips=13):
        # x: (B*T, C, H, W)
        bt, c, h, w = x.size()
        b = bt // clips
        x = x.view(b, clips, c, h, w).contiguous()

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # not shift

        return out.view(b * clips, c, h, w).contiguous()
        # out: (B*T, C, H, W)
