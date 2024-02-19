#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 18:50:02
"""

import numpy as np
import torch

import config
from speaker_verification import EcapaTdnnVerificator

verificator: EcapaTdnnVerificator

def load_verificator():
    global verificator
    verificator = EcapaTdnnVerificator(cpu=config.speaker_verificate_cpu)
    return verificator


def verify_speakers(
    audio: np.ndarray,
) -> str:
    lib_feat, lib_labels = get_lib_feat_and_labels()
    feat = verificator.gen_feat(audio)
    score = verificator.calc_score_batch(feat.unsqueeze(0), lib_feat)
    score = score[0]
    max_idx = torch.argmax(score)
    max_score = score[max_idx]
    if max_score < config.speaker_verificate_score_threshold:
        return create_new_label(feat)
    label = lib_labels[max_idx]
    return label


def get_lib_feat_and_labels() -> tuple[torch.Tensor, list[str]]:
    # TODO
    return torch.Tensor([]), []


def create_new_label(feature: torch.Tensor) -> str:
    # TODO
    return "new_label"
