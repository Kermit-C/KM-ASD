#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from torchvision import transforms

# 视频训练集的数据增强，包括转换为张量和归一化
video_train = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.3729, 0.2850, 0.2439), (0.2286, 0.2008, 0.1911)),
    ]
)

# 视频验证集的数据增强，包括转换为张量和归一化
video_val = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.3729, 0.2850, 0.2439), (0.2286, 0.2008, 0.1911)),
    ]
)
