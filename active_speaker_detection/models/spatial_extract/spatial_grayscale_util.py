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
    spatial_grayscale[:, :, positions[:, 0, 0] : positions[:, 0, 2], positions[:, 0, 1] : positions[:, 0, 3]] = 1
    spatial_grayscale[:, :, positions[:, 1, 0] : positions[:, 1, 2], positions[:, 1, 1] : positions[:, 1, 3]] = 1
    return spatial_grayscale
