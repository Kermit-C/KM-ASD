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
from torch_geometric.nn import BatchNorm

from active_speaker_detection.utils.vf_util import cosine_similarity


class GraphFcNet(nn.Module):

    def __init__(
        self,
        in_a_channels,
        in_v_channels,
        in_vf_channels,
        channels,
    ):
        super().__init__()

        self.in_a_channels = in_a_channels
        self.in_v_channels = in_v_channels
        self.in_vf_channels = in_vf_channels
        self.channels = channels

        self.layer_0_a = nn.Linear(in_a_channels, channels)
        self.layer_0_v = nn.Linear(in_v_channels, channels)
        self.av_fusion = nn.Linear(channels * 2, channels)
        self.batch_0 = BatchNorm(channels)

        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x = data.x
        audio_node_mask = []
        for mask in data.audio_node_mask:
            audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        graph_feats = torch.zeros(x.size(0), self.channels, dtype=x.dtype).to(x.device)
        audio_feats = x[:, 0, : self.in_a_channels][audio_node_mask]
        video_audio_feats = x[:, 0, : self.in_a_channels][video_node_mask]
        video_feats = x[:, 1, : self.in_v_channels][video_node_mask]

        audio_vf_emb = (
            x[:, 2, : self.in_vf_channels][video_node_mask] if x.size(1) > 2 else None
        )
        video_vf_emb = (
            x[:, 3, : self.in_vf_channels][video_node_mask] if x.size(1) > 3 else None
        )
        if audio_vf_emb is not None and video_vf_emb is not None:
            # sim 的维度是 (B, )
            sim = cosine_similarity(audio_vf_emb, video_vf_emb)
            video_audio_feats = video_audio_feats * sim.unsqueeze(1)

        graph_feats[audio_node_mask] = self.layer_0_a(audio_feats)
        graph_feats[video_node_mask] = self.layer_0_v(video_feats)
        graph_feats[video_node_mask] = self.av_fusion(
            torch.cat([video_audio_feats, graph_feats[video_node_mask]], dim=1)
        )
        graph_feats = self.batch_0(graph_feats)
        graph_feats = self.relu(graph_feats)

        out = self.fc(graph_feats)

        return out
