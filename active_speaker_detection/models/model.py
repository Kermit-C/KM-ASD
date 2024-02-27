#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""


import torch
import torch.nn as nn
import torch.nn.parameter

from active_speaker_detection.models.two_stream_light_net import get_light_encoder
from active_speaker_detection.models.two_stream_resnet_tsm_net import (
    get_resnet_tsm_encoder,
)

from .graph_all_edge_net import GraphAllEdgeNet
from .two_stream_resnet_net import get_resnet_encoder

############### 以下是模型的加载权重 ###############

def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)
    return


############### 以下是模型的工厂函数 ###############


def get_backbone(
    encoder_type: str,
    graph_type: str,
    encoder_enable_vf: bool,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    encoder_train_weights=None,
    train_weights=None,
):
    # 加载 encoder
    if encoder_type == "R3D18":
        encoder = get_resnet_encoder(
            "R3D18",
            encoder_enable_vf,
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "R3D50":
        encoder = get_resnet_encoder(
            "R3D50",
            encoder_enable_vf,
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "LIGHT":
        encoder = get_light_encoder(encoder_train_weights, encoder_enable_vf)
    elif encoder_type == "RES18_TSM":
        encoder = get_resnet_tsm_encoder(
            "resnet18",
            encoder_enable_vf,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RES50_TSM":
        encoder = get_resnet_tsm_encoder(
            "resnet50",
            encoder_enable_vf,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # 加载整个图模型
    if graph_type == "GraphAllEdgeNet":
        model = GraphAllEdgeNet(encoder, 128)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # 加载模型权重
    # 先加载整个的，如果有单独的 encoder 的权重，再加载 encoder 的权重
    if train_weights is not None:
        _load_weights_into_model(model, train_weights)
        model.eval()
    if encoder_train_weights is not None:
        _load_weights_into_model(encoder, encoder_train_weights)

    return model, encoder
