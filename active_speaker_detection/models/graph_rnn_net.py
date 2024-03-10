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
        self.channels = channels
        self.max_context = max_context

        self.layer_0_a = nn.Linear(in_a_channels, channels)
        self.layer_0_v = nn.Linear(in_v_channels, channels)
        self.av_fusion = nn.Linear(2 * channels, channels)
        self.batch_0_a = BatchNorm(channels)
        self.batch_0_v = BatchNorm(channels)

        if type == "GRU":
            self.rnn = nn.GRU(
                channels,
                channels,
                num_layers=2,
                dropout=0.2,
                bidirectional=False,
                batch_first=True,
            )
        elif type == "LSTM":
            self.rnn = nn.LSTM(
                channels,
                channels,
                num_layers=2,
                dropout=0.2,
                bidirectional=False,
                batch_first=True,
            )
        else:
            raise ValueError("RNN type must be GRU or LSTM")

        # 分类器
        self.fc_a = nn.Linear(channels, 2)
        self.fc_v = nn.Linear(channels, 2)
        self.fc = nn.Linear(channels, 2)

        # 共享
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dropout_edge = 0

    def forward(self, data):
        x, edge_index, edge_attr, y = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.y,
        )
        entities = y[:, -1]
        audio_node_mask = []
        for mask in data.audio_node_mask:
            audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        audio_feats = x[:, 0, : self.in_a_channels]
        video_feats = x[:, 1, : self.in_v_channels]
        audio_emb = self.batch_0_a(self.layer_0_a(audio_feats))
        video_emb = self.batch_0_v(self.layer_0_v(video_feats))
        audio_vf_emb = x[:, 2, : self.in_vf_channels] if x.size(1) > 2 else None
        video_vf_emb = x[:, 3, : self.in_vf_channels] if x.size(1) > 3 else None
        if audio_vf_emb is not None and video_vf_emb is not None:
            # sim 的维度是 (B, )
            sim = cosine_similarity(audio_vf_emb, video_vf_emb)
            audio_emb = audio_emb * sim.unsqueeze(1)
        graph_feats = self.av_fusion(torch.cat([audio_emb, video_emb], dim=1))
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
        audio_out = self.fc_a(audio_emb[audio_node_mask])
        video_out = self.fc_v(video_emb[video_node_mask])

        return out, audio_out, video_out
