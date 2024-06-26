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
    value: float = 1,
) -> torch.Tensor:
    """生成空间位置关系灰度图"""
    spatial_grayscale = torch.zeros(
        3, target_w, target_h, device=device, dtype=torch.float32
    )
    for x1, y1, x2, y2 in positions:
        x1 = int(x1 * target_w)
        y1 = int(y1 * target_h)
        x2 = int(x2 * target_w)
        y2 = int(y2 * target_h)
        spatial_grayscale[:, x1:x2, y1:y2] = value
    return spatial_grayscale


def batch_create_spatial_grayscale(
    positions: torch.Tensor,  # [batch, 2, 4]
    target_w: int,
    target_h: int,
    values: torch.Tensor,  # [batch,]
) -> torch.Tensor:
    """批量生成空间位置关系灰度图"""
    spatial_grayscale = torch.zeros(
        positions.size(0),
        3,
        target_h,
        target_w,
        device=positions.device,
        dtype=torch.float32,
    )
    positions = positions[:, :, :] * torch.tensor(
        [target_h, target_w, target_h, target_w],
        device=positions.device,
        dtype=positions.dtype,
    ).view(1, 1, 4)
    positions = positions.int()
    for i in range(positions.size(0)):
        x1, y1, x2, y2 = positions[i, 0, :]
        spatial_grayscale[i, 0, x1:x2, y1:y2] = values[i]
        x1, y1, x2, y2 = positions[i, 1, :]
        spatial_grayscale[i, 1, x1:x2, y1:y2] = values[i]
    return spatial_grayscale
