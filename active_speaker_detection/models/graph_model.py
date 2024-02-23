#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.parameter
from torch.nn import functional as F
from torch_geometric.nn import EdgeConv

from active_speaker_detection.models.light_two_stream_net import get_light_encoder

from .graph_layouts import generate_av_mask
from .resnet_two_stream_net import get_resnet_encoder


class LinearPathPreact(nn.Module):
    """EdgeConv 的线性路径预激活模块"""

    def __init__(self, in_channels, hidden_channels):
        super(LinearPathPreact, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x1 = self.fc1(x1)

        x2 = self.bn2(x1)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)

        return x2


class GraphNet(nn.Module):
    """图神经网络"""

    def __init__(self, encoder, filter_size):
        super().__init__()

        self.encoder = encoder
        self.reduction_v_vfal = nn.Linear(128 * 3, 128)

        self.edge_spatial_1 = EdgeConv(LinearPathPreact(128 * 2, filter_size))
        self.edge_spatial_2 = EdgeConv(LinearPathPreact(filter_size * 2, filter_size))
        self.edge_spatial_3 = EdgeConv(LinearPathPreact(filter_size * 2, filter_size))
        self.edge_spatial_4 = EdgeConv(LinearPathPreact(filter_size * 2, filter_size))

        self.edge_temporal_1 = EdgeConv(LinearPathPreact(filter_size * 2, filter_size))
        self.edge_temporal_2 = EdgeConv(LinearPathPreact(filter_size * 2, filter_size))
        self.edge_temporal_3 = EdgeConv(LinearPathPreact(filter_size * 2, filter_size))
        self.edge_temporal_4 = EdgeConv(LinearPathPreact(filter_size * 2, filter_size))

        self.fc = nn.Linear(filter_size, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        data,
        ctx_size,
        audio_size: Tuple[int, int, int],
        vfal_size: Tuple[int, int, int],
    ):
        x, x2, joint_edge_index, _ = data.x, data.x2, data.edge_index, data.batch
        spatial_edge_index = joint_edge_index[0]
        temporal_edge_index = joint_edge_index[1]

        # 生成音频和视频的 mask
        audio_mask, video_mask = generate_av_mask(ctx_size, x.size(0))

        # 音频和视频的特征提取
        audio_feats = self.encoder.forward_audio(x[audio_mask], audio_size)
        video_feats = self.encoder.forward_video(x[video_mask])
        vfal_a_feats, vfal_v_feats = self.encoder.forward_vfal(
            x2, video_feats, vfal_size
        )

        # 降维，降到节点特征的维度 128
        audio_feats = self.relu(self.encoder.reduction_a(audio_feats))
        video_feats = self.relu(self.encoder.reduction_v(video_feats))
        # video_feats = self.relu(
        #     self.reduction_v_vfal(
        #         torch.cat(
        #             [
        #                 video_feats,
        #                 # 重复 vfal_a_feats，使得 batch 维度相同
        #                 torch.cat(
        #                     [
        #                         torch.stack(
        #                             [vfal_a]
        #                             * (video_feats.size(0) // vfal_a_feats.size(0)),
        #                             dim=0,
        #                         )
        #                         for vfal_a in vfal_a_feats
        #                     ],
        #                     dim=0,
        #                 ),
        #                 vfal_v_feats,
        #             ],
        #             dim=1,
        #         )
        #     )
        # )

        # 重建交错张量
        graph_feats = torch.zeros(
            (x.size(0), 128), device=audio_feats.get_device(), dtype=audio_feats.dtype
        )
        graph_feats[audio_mask] = audio_feats
        graph_feats[video_mask] = video_feats

        # 辅助监督
        audio_out = self.encoder.fc_aux_a(graph_feats[audio_mask])
        video_out = self.encoder.fc_aux_v(graph_feats[video_mask])

        # 有残差的图神经网络
        graph_feats_1s = self.edge_spatial_1(graph_feats, spatial_edge_index)
        graph_feats_1st = self.edge_temporal_1(graph_feats_1s, temporal_edge_index)

        graph_feats_2s = self.edge_spatial_2(graph_feats_1st, spatial_edge_index)
        graph_feats_2st = self.edge_temporal_2(graph_feats_2s, temporal_edge_index)
        graph_feats_2st = graph_feats_2st + graph_feats_1st

        graph_feats_3s = self.edge_spatial_3(graph_feats_2st, spatial_edge_index)
        graph_feats_3st = self.edge_temporal_3(graph_feats_3s, temporal_edge_index)
        graph_feats_3st = graph_feats_3st + graph_feats_2st

        graph_feats_4s = self.edge_spatial_4(graph_feats_3st, spatial_edge_index)
        graph_feats_4st = self.edge_temporal_4(graph_feats_4s, temporal_edge_index)
        graph_feats_4st = graph_feats_4st + graph_feats_3st

        return (
            self.fc(graph_feats_4st),
            audio_out,
            video_out,
            vfal_a_feats,
            vfal_v_feats,
        )


############### 以下是模型的加载权重 ###############

def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file))
    return


############### 以下是模型的工厂函数 ###############


def get_backbone(
    encoder_type: str,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    vfal_ecapa_pretrain_weights=None,
    encoder_train_weights=None,
    train_weights=None,
):
    if encoder_type == "R3D18":
        encoder = get_resnet_encoder(
            "R3D18",
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "R3D50":
        encoder = get_resnet_encoder(
            "R3D50",
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "LIGHT":
        encoder = get_light_encoder(encoder_train_weights)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    model = GraphNet(encoder, 128)
    if train_weights is not None:
        _load_weights_into_model(model, train_weights)

    return model, encoder
