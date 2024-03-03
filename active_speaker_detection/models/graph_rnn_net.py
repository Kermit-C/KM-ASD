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


class GraphRnnNet(nn.Module):

    def __init__(
        self,
        type: str,
        in_a_channels,
        in_v_channels,
        in_vf_channels,
        channels,
        max_context: int = 3,
    ):
        super().__init__()

        self.in_a_channels = in_a_channels
        self.in_v_channels = in_v_channels
        self.in_vf_channels = in_vf_channels
        self.max_context = max_context

        self.av_fusion = nn.Linear(in_a_channels + in_v_channels, channels)
        self.batch_0 = BatchNorm(channels)

        if type == "GRU":
            self.rnn = nn.GRU(
                channels,
                channels,
                num_layers=2,
                dropout=0.1,
                bidirectional=False,
                batch_first=True,
            )
        elif type == "LSTM":
            self.rnn = nn.LSTM(
                channels,
                channels,
                num_layers=2,
                dropout=0.1,
                bidirectional=False,
                batch_first=True,
            )
        else:
            raise ValueError("RNN type must be GRU or LSTM")

        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dropout_edge = 0.2

    def forward(self, data):
        x, edge_index, edge_attr, y = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.y,
        )
        entities = y[:, -1]

        audio_feats = x[:, 0, : self.in_a_channels].squeeze(1)
        video_feats = x[:, 1, : self.in_v_channels].squeeze(1)
        audio_vf_emb = (
            x[:, 2, : self.in_vf_channels].squeeze(1) if x.size(1) > 2 else None
        )
        video_vf_emb = (
            x[:, 3, : self.in_vf_channels].squeeze(1) if x.size(1) > 3 else None
        )
        if audio_vf_emb is not None and video_vf_emb is not None:
            # sim 的维度是 (B, )
            sim = cosine_similarity(audio_vf_emb, video_vf_emb)
            audio_feats = audio_feats * sim.unsqueeze(1)
        graph_feats = self.av_fusion(torch.cat([audio_feats, video_feats], dim=1))
        graph_feats = self.batch_0(graph_feats)
        graph_feats = self.relu(graph_feats)

        # entity 去重
        entity_list = list(set(entities))
        entity_mask_list = [entities == entity for entity in entity_list]
        max_entity_mask_len = max([entities[mask].size(0) for mask in entity_mask_list])

        # 构造 rnn 输入
        rnn_x = torch.zeros(
            len(entity_list), max_entity_mask_len, graph_feats.size(1)
        ).to(graph_feats.device)
        for i, mask in enumerate(entity_mask_list):
            rnn_x[i, : entities[mask].size(0)] = graph_feats[mask]
            # TODO: 利用 edge_index 融合上下文信息

        rnn_x, _ = self.rnn(rnn_x)

        # 构造图的输出
        graph_feats_out = torch.zeros(
            graph_feats.size(0), rnn_x.size(2), dtype=rnn_x.dtype
        ).to(graph_feats.device)
        for i, mask in enumerate(entity_mask_list):
            graph_feats_out[mask] = rnn_x[i, : entities[mask].size(0)]

        out = self.fc(graph_feats_out)

        return out
