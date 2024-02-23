#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.parameter
from torch.nn import functional as F

from active_speaker_detection.models.vfal.vfal_ecapa_tdnn_net import get_ecapa_net
from active_speaker_detection.models.vfal.vfal_sl_encoder import VfalSlEncoder


class VfalTwoStreamNet(nn.Module):
    """
    音脸关系的双流网络
    """

    def __init__(self):
        super().__init__()
        self.vfal_ecapa = get_ecapa_net()
        self.vfal_encoder = VfalSlEncoder(
            voice_size=192, face_size=512, embedding_size=128
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        a_vfal: torch.Tensor,
        video_feat: torch.Tensor,
        vfal_size: Tuple[int, int, int],
    ):
        a_vfal = torch.squeeze(a_vfal[:, : vfal_size[1], : vfal_size[2]], dim=1)
        a_vfal = self.vfal_ecapa(
            a_vfal, torch.ones(a_vfal.size(0), dtype=torch.float32)
        )
        a_vfal = torch.squeeze(a_vfal, dim=1)
        a_vfal, v_vfal = self.vfal_encoder(a_vfal, video_feat)
        return a_vfal, v_vfal


############### 以下是模型的加载权重 ###############


def _load_vfal_weights_into_model(model: nn.Module, vfal_ecapa_pretrain_weights):
    """加载音脸关系预训练权重"""
    checkpoint = torch.load(vfal_ecapa_pretrain_weights)

    model.vfal_ecapa.load_state_dict(checkpoint["model"], strict=True)  # type: ignore
    # 固定参数
    model.vfal_ecapa.eval()  # type: ignore

    # TMP: 加载 vfal encoder 的预训练权重
    vfal_encoder_checkpoint = torch.load(
        "/hdd1/ckm/tools/sl_icmr2022/outputs/sl_project/SL/auc[86.91,86.66]_ms[86.43,86.59]_map[7.09,7.90].pkl"
    )
    model.vfal_encoder.load_state_dict(vfal_encoder_checkpoint["model"], strict=True)  # type: ignore
    model.vfal_encoder.eval()  # type: ignore

    print("loaded vfal ws")
    return


############### 以下是模型的工厂函数 ###############


def get_vfal_encoder(
    vfal_ecapa_pretrain_weights=None,
):
    encoder = VfalTwoStreamNet()
    if vfal_ecapa_pretrain_weights is not None:
        _load_vfal_weights_into_model(encoder, vfal_ecapa_pretrain_weights)
    return encoder
