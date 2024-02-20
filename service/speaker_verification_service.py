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
from store.local_store import LocalStore
from utils.hash_util import calculate_md5
from utils.uuid_util import get_uuid

from .store.speaker_verification_store import SpeakerVerificationStore

verificator: EcapaTdnnVerificator
speaker_verification_store: SpeakerVerificationStore


def load_verificator():
    global verificator
    verificator = EcapaTdnnVerificator(cpu=config.speaker_verificate_cpu)
    return verificator


def load_speaker_verification_store():
    global speaker_verification_store
    speaker_verification_store = SpeakerVerificationStore(
        LocalStore.create, max_face_count=1000
    )
    # TODO: 实现注册声纹
    return speaker_verification_store


def verify_speakers(
    audio: np.ndarray,
) -> str:
    lib_feat, lib_labels = get_lib_feat_and_labels()
    feat = verificator.gen_feat(audio)
    if lib_feat.size(0) == 0:
        return create_new_label(feat)
    score = verificator.calc_score_batch(feat.unsqueeze(0), lib_feat)
    score = score[0]
    max_idx = torch.argmax(score)
    max_score = score[max_idx]
    if max_score < config.speaker_verificate_score_threshold:
        return create_new_label(feat)
    label = lib_labels[max_idx]
    return label


def get_lib_feat_and_labels() -> tuple[torch.Tensor, list[str]]:
    # TODO: 存在上一次的新 label 未保存完，这里又存了新的 label 的问题，考虑用 label 关联解决
    labels, feats = speaker_verification_store.get_all_feats()
    lib_labels = labels
    lib_feat = torch.stack(feats)
    return lib_feat, lib_labels


def create_new_label(feature: torch.Tensor) -> str:
    label = str(calculate_md5(get_uuid()) % 1000000000)
    speaker_verification_store.save_feat(label, feature)
    return label
