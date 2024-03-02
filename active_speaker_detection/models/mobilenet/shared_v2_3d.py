#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-03-02 14:03:54
"""

import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(
            inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False
        ),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
