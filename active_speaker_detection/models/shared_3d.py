#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@description: 
@author: chenkeming
@date: Do not edit
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock3D(nn.Module):
    """
    3D ResNet 18 的基本块
    和 2D ResNet 的基本块相比，变成了 3D 卷积
    """
    expansion = 1

    def __init__(
        self,
        in_planes: int,  # 输入通道数
        planes: int,  # 隐层通道数
        stride=1,  # 卷积步长
        downsample: Optional[nn.Module] = None,  # 下采样块，用于残差连接时的维度匹配
    ):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    """
    3D ResNet 50 的瓶颈块
    和 2D ResNet 的瓶颈块相比，变成了 3D 卷积
    """
    expansion = 4

    def __init__(
        self,
        in_planes: int,  # 输入通道数
        planes: int,  # 隐层通道数
        stride: int = 1,  # 卷积步长
        downsample: Optional[nn.Module] = None,  # 下采样块，用于残差连接时的维度匹配
    ):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
