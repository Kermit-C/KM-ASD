#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-16 14:36:50
"""

import torch
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN


def get_ecapa_net(
    n_mels=80,
    channels=[1024, 1024, 1024, 1024, 3072],
    kernel_sizes=[5, 3, 3, 3, 1],
    dilations=[1, 2, 3, 4, 1],
    attention_channels=128,
    lin_neurons=192,
) -> torch.nn.Module:
    model = ECAPA_TDNN(
        input_size=n_mels,
        channels=channels,
        kernel_sizes=kernel_sizes,
        dilations=dilations,
        attention_channels=attention_channels,
        lin_neurons=lin_neurons,
    )
    return model
