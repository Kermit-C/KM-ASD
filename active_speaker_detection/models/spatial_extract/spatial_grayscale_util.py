#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-28 17:13:21
"""

import torch


def create_spatial_grayscale(
    positions: list[tuple[float, float, float, float]],
    device: torch.device,
    target_w: int,
    target_h: int,
    target_channels: int = 3,
) -> torch.Tensor:
    """生成空间位置关系灰度图"""
    spatial_grayscale = torch.zeros(target_channels, target_w, target_h, device=device)
    for x1, y1, x2, y2 in positions:
        x1 = int(x1 * target_w)
        y1 = int(y1 * target_h)
        x2 = int(x2 * target_w)
        y2 = int(y2 * target_h)
        spatial_grayscale[:, x1:x2, y1:y2] = 1
    return spatial_grayscale


def batch_create_spatial_grayscale(
    positions: torch.Tensor,  # [batch, 2, 4]
    target_w: int,
    target_h: int,
    target_channels: int = 3,
) -> torch.Tensor:
    """批量生成空间位置关系灰度图"""
    spatial_grayscale = torch.zeros(
        positions.size(0), target_channels, target_w, target_h, device=positions.device
    )
    positions = positions[:, :, :] * torch.tensor(
        [target_w, target_h, target_w, target_h], device=positions.device
    ).view(1, 1, 4)
    positions = positions.int()
    for i in range(positions.size(0)):
        x1, y1, x2, y2 = positions[i, 0, :]
        spatial_grayscale[i, :, x1:x2, y1:y2] = 1
        x1, y1, x2, y2 = positions[i, 1, :]
        spatial_grayscale[i, :, x1:x2, y1:y2] = 1
    return spatial_grayscale
