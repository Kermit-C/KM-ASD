#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-16 14:55:49
"""

import torch


class VfalSlEncoder(torch.nn.Module):
    def __init__(self, voice_size=192, face_size=512, embedding_size=128, shared=True):
        super(VfalSlEncoder, self).__init__()
        # input->drop-fc256-relu-[fc256-relu-fc128]
        mid_dim = 256

        def create_front(input_size):
            return torch.nn.Sequential(
                torch.nn.Dropout(0.4),
                torch.nn.Linear(input_size, mid_dim),
                torch.nn.ReLU(),
            )

        def create_rare():
            return torch.nn.Sequential(
                torch.nn.Dropout(0.4),
                torch.nn.Linear(mid_dim, mid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(mid_dim, embedding_size),
            )

        face_rare = create_rare()
        if shared:
            voice_rare = face_rare
        else:
            voice_rare = create_rare()

        self.face_encoder = torch.nn.Sequential(create_front(face_size), face_rare)
        self.voice_encoder = torch.nn.Sequential(create_front(voice_size), voice_rare)

    def forward(self, voice_data, face_data):
        v_emb = self.voice_encoder(voice_data)
        f_emb = self.face_encoder(face_data)
        return v_emb, f_emb
