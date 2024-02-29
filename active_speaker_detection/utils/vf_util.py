#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-29 15:46:29
"""

import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(a.shape) == 2
    assert a.shape == b.shape

    ab = torch.sum(a * b, dim=1)
    # (batch_size,)

    a_norm = torch.sqrt(torch.sum(a * a, dim=1))
    b_norm = torch.sqrt(torch.sum(b * b, dim=1))
    cosine = ab / (a_norm * b_norm)
    # [-1,1]
    prob = (cosine + 1) / 2.0
    # [0,1]
    return prob
