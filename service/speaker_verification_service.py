#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 18:50:02
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
import torchaudio

import config
from speaker_verification import EcapaTdnnVerificator
from store.local_store import LocalStore
from utils.hash_util import calculate_md5
from utils.uuid_util import get_uuid

from .store.speaker_verification_store import SpeakerVerificationStore

verificator: Optional[EcapaTdnnVerificator] = None
speaker_verification_store: SpeakerVerificationStore


def load_verificator():
    global verificator
    if verificator is not None:
        return verificator
    verificator = EcapaTdnnVerificator(cpu=config.speaker_verificate_cpu)
    return verificator


def load_speaker_verification_store():
    global speaker_verification_store
    # TODO: 需要使用 RedisStore，保证多个实例的数据一致性
    speaker_verification_store = SpeakerVerificationStore(
        LocalStore.create, max_face_count=1000
    )
    register_speakers()
    return speaker_verification_store


def register_speakers():
    if not os.path.exists(config.speaker_verificate_register_path):
        return
    for root, dirs, files in os.walk(config.speaker_verificate_register_path):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                label = ".".join(os.path.basename(file).split(".")[:-1])
                audio = capture_audio(audio_path)
                register_speaker(audio.numpy(), label)
                logging.info(f"Register speaker {label} successfully")


def register_speaker(audio: np.ndarray, label: str):
    global verificator
    if verificator is None:
        verificator = load_verificator()
    feat = verificator.gen_feat(audio)
    speaker_verification_store.save_feat(label, feat)


def verify_speakers(
    audio: np.ndarray,
) -> str:
    if verificator is None:
        raise ValueError("Verificator is not loaded")
    lib_feat, lib_labels = get_lib_feat_and_labels()
    feat = verificator.gen_feat(audio)
    if lib_feat.size(0) == 0:
        return create_new_label(feat)
    score = verificator.calc_score_batch(feat.unsqueeze(0), lib_feat)
    score = score[0]
    max_idx = np.argmax(score)
    max_score = score[max_idx]
    if max_score < config.speaker_verificate_score_threshold:
        return create_new_label(feat)
    label = lib_labels[max_idx]
    return label


def get_lib_feat_and_labels() -> tuple[torch.Tensor, list[str]]:
    # TODO: 存在上一次的新 label 未保存完，这里又存了新的 label 的问题，考虑用 label 关联解决
    labels, feats = speaker_verification_store.get_all_feats()
    lib_labels = labels
    lib_feat = torch.stack(feats) if len(feats) > 0 else torch.Tensor()
    return lib_feat, lib_labels


def create_new_label(feature: torch.Tensor) -> str:
    label = str(calculate_md5(get_uuid()) % 1000000000)
    speaker_verification_store.save_feat(label, feature)
    return label


def capture_audio(audio_path: str) -> torch.Tensor:
    # 读取音频文件
    # audio: (num_channels, num_samples)
    audio: torch.Tensor
    sample_rate: int
    audio, sample_rate = torchaudio.load(audio_path)  # type: ignore
    if sample_rate != config.speaker_verificate_sample_rate:
        # 重采样
        audio = torchaudio.transforms.Resample(
            sample_rate, config.speaker_verificate_sample_rate
        )(audio)
    # 取单声道 (num_samples,)
    audio = audio[0]  # 采样值在 -1 到 1 之间
    # audio: (num_samples,)
    return audio
