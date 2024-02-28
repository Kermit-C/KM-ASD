#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.parameter

from .graph_all_edge_net import GraphAllEdgeNet
from .graph_gat_edge_net import GraphGatEdgeNet
from .graph_gated_edge_net import GraphGatedEdgeNet
from .spatial_extract.spatial_mobilenet_net import get_spatial_mobilenet_net
from .two_stream_light_net import get_light_encoder
from .two_stream_resnet_net import get_resnet_encoder
from .two_stream_resnet_tsm_net import get_resnet_tsm_encoder


class MyModel(nn.Module):
    def __init__(self, encoder, graph_net):
        super().__init__()

        self.encoder = encoder
        self.graph_net = graph_net

    def forward(
        self,
        data,
        audio_size: Optional[tuple[int, int, int]] = None,
    ):
        x = data.x
        if len(x.shape) > 3:
            # 从数据中提取音频和视频特征
            assert audio_size is not None
            audio_data = (
                x[:, 0, 0, 0, : audio_size[1], : audio_size[2]]
                .unsqueeze(1)
                .unsqueeze(1)
            )
            video_data = x[:, 1, :, :, :, :].unsqueeze(1)
            audio_feats, video_feats, audio_out, video_out, _, vf_a_emb, vf_v_emb = (
                self.encoder(audio_data, video_data)
            )
            data.x = torch.stack([audio_feats, video_feats], dim=1)
        else:
            # 输入的就是 encoder 出来的 128 维特征
            audio_out = None
            video_out = None
            vf_a_emb = None
            vf_v_emb = None

        graph_out = self.graph_net(data)

        return graph_out, audio_out, video_out, vf_a_emb, vf_v_emb


############### 以下是模型的加载权重 ###############

def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)


############### 以下是模型的工厂函数 ###############


def get_backbone(
    encoder_type: str,
    graph_type: str,
    encoder_enable_vf: bool,
    graph_enable_spatial: bool,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    spatial_pretrained_weights=None,
    encoder_train_weights=None,
    train_weights=None,
):
    encoder = get_encoder(
        encoder_type,
        encoder_enable_vf,
        video_pretrained_weigths,
        audio_pretrained_weights,
        encoder_train_weights,
    )
    graph_net = get_graph(graph_type, graph_enable_spatial, spatial_pretrained_weights)
    model = MyModel(encoder, graph_net)

    # 加载模型权重
    if train_weights is not None:
        _load_weights_into_model(model, train_weights)
        print("loaded model weights")
        model.eval()
        # 先加载整个的，如果有单独的 encoder 的权重，再覆盖一次 encoder 的权重
        if encoder_train_weights is not None:
            _load_weights_into_model(encoder, encoder_train_weights)
            print("loaded encoder weights")

    return model, encoder


def get_encoder(
    encoder_type: str,
    encoder_enable_vf: bool,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    encoder_train_weights=None,
):
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

    return encoder


def get_graph(
    graph_type: str,
    graph_enable_spatial: bool,
    spatial_pretrained_weights: Optional[str] = None,
):
    spatial_feature_dim = 64
    spatial_net = (
        get_spatial_mobilenet_net(spatial_feature_dim, spatial_pretrained_weights)
        if graph_enable_spatial
        else None
    )
    if graph_type == "GraphAllEdgeNet":
        graph_net = GraphAllEdgeNet(128, spatial_net, spatial_feature_dim)
    elif graph_type == "GraphGatEdgeNet":
        graph_net = GraphGatEdgeNet(128, spatial_net, spatial_feature_dim)
    elif graph_type == "GraphGatedEdgeNet":
        graph_net = GraphGatedEdgeNet(128, spatial_net, spatial_feature_dim)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return graph_net
