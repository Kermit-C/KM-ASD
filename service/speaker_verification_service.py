#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 18:50:02
"""

import multiprocessing as mp
import os
from multiprocessing.pool import Pool
from typing import Optional, Union

import numpy as np
import torch
import torchaudio

import config
from speaker_verification import EcapaTdnnVerificator
from store.local_store import LocalStore
from utils.hash_util import calculate_md5
from utils.logger_util import infer_logger, ms_logger
from utils.uuid_util import get_uuid

from .store.speaker_verification_store import SpeakerVerificationStore

# 主进程中的全局变量
verificator_pool: Optional[Pool] = None
speaker_verification_store: SpeakerVerificationStore

# 进程池每个进程中的全局变量
verificator: Optional[EcapaTdnnVerificator] = None


def load_verificator():
    global verificator_pool
    if verificator_pool is None:
        mp.set_start_method("spawn", True)
        verificator_pool = mp.Pool(
            config.model_service_server_speaker_verificate_max_workers,
            initializer=init_verificator_pool_process,
        )
    ms_logger.info("speaker verificator pool loaded")
    return verificator_pool


def load_speaker_verification_store():
    global speaker_verification_store
    # TODO: 需要使用 RedisStore，保证多个实例的数据一致性
    speaker_verification_store = SpeakerVerificationStore(
        LocalStore.create, max_face_count=1000
    )
    register_speakers()
    return speaker_verification_store


# 初始化进程池的进程
def init_verificator_pool_process():
    global verificator
    if verificator is not None:
        return verificator
    verificator = EcapaTdnnVerificator(cpu=config.speaker_verificate_cpu)
    ms_logger.info("speaker verificator worker loaded")
    return verificator


# 以下是进程池中的函数
def verificator_gen_feat(audio: Union[str, np.ndarray]) -> torch.Tensor:
    if verificator is None:
        raise ValueError("Verificator is not loaded")
    return verificator.gen_feat(audio)


def verificator_calc_score_batch(emb1: torch.Tensor, emb2: torch.Tensor) -> np.ndarray:
    if verificator is None:
        raise ValueError("Verificator is not loaded")
    return verificator.calc_score_batch(emb1, emb2)


# 以下是主进程中的函数

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
                ms_logger.info(f"register speaker {label} successfully")


def register_speaker(audio: np.ndarray, label: str):
    global verificator_pool
    if verificator_pool is None:
        verificator_pool = load_verificator()
    feat = verificator_pool.apply(verificator_gen_feat, (audio,))
    speaker_verification_store.save_feat(label, feat)


def verify_speakers(
    audio: np.ndarray,
) -> str:
    if verificator_pool is None:
        raise ValueError("Verificator pool is not loaded")
    lib_feat, lib_labels = get_lib_feat_and_labels()
    if lib_feat.size(0) == 0:
        return ""
    feat = verificator_pool.apply(verificator_gen_feat, (audio,))
    score = verificator_pool.apply(
        verificator_calc_score_batch, (feat.unsqueeze(0), lib_feat)
    )
    score = score[0]
    max_idx = np.argmax(score)
    max_score = score[max_idx]
    if max_score < config.speaker_verificate_score_threshold:
        return ""
    label = lib_labels[max_idx]
    return label


def get_lib_feat_and_labels() -> tuple[torch.Tensor, list[str]]:
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
