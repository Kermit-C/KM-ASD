#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-03-02 02:20:59
"""

import torch
import torch.nn as nn


class SpatialNet(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super(SpatialNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, kernel_size=5)
        self.conv2 = nn.Conv2d(18, 48, kernel_size=5)
        self.fc1 = nn.Linear(48 * 5 * 5, 360)
        self.fc2 = nn.Linear(360, 120)
        self.fc3 = nn.Linear(120, feature_dim)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool2d(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool2d(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


############### 以下是模型的工厂函数 ###############


def get_spatial_net(
    feature_dim: int,  # 输出特征维度
):
    model = SpatialNet(feature_dim)
    return model
