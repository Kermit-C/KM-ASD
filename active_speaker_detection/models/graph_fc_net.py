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
        self.batch_0_a = BatchNorm(channels)
        self.batch_0_v = BatchNorm(channels)

        # 分类器
        self.fc_a = nn.Linear(channels, 2)
        self.fc_v = nn.Linear(channels, 2)
        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x = data.x
        audio_node_mask = []
        if type(data.audio_node_mask[0]) == bool:
            audio_node_mask = data.audio_node_mask
        else:
            for mask in data.audio_node_mask:
                audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        graph_feats = torch.zeros(x.size(0), self.channels, dtype=x.dtype).to(x.device)
        audio_feats = x[audio_node_mask][:, 0, : self.in_a_channels]
        video_audio_feats = x[video_node_mask][:, 0, : self.in_a_channels]
        video_feats = x[video_node_mask][:, 1, : self.in_v_channels]

        audio_vf_emb = (
            x[video_node_mask][:, 2, : self.in_vf_channels] if x.size(1) > 2 else None
        )
        video_vf_emb = (
            x[video_node_mask][:, 3, : self.in_vf_channels] if x.size(1) > 3 else None
        )
        if audio_vf_emb is not None and video_vf_emb is not None:
            # sim 的维度是 (B, )
            sim = cosine_similarity(audio_vf_emb, video_vf_emb)
            video_audio_feats = video_audio_feats * sim.unsqueeze(1)

        graph_feats[audio_node_mask] = self.batch_0_a(self.layer_0_a(audio_feats))
        graph_feats[video_node_mask] = self.batch_0_v(
            self.av_fusion(
                torch.cat(
                    [
                        self.relu(self.layer_0_a(video_audio_feats)),
                        self.relu(self.layer_0_v(video_feats)),
                    ],
                    dim=1,
                )
            )
        )
        graph_feats = self.relu(graph_feats)

        out = self.fc(graph_feats)
        audio_out = self.fc_a(graph_feats[audio_node_mask])
        video_out = self.fc_v(graph_feats[video_node_mask])

        return out, audio_out, video_out
