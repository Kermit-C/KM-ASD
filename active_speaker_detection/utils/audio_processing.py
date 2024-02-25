#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import numpy as np
import python_speech_features
import torch
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization


def generate_mel_spectrogram(audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
    """计算音频片段的梅尔频谱图，维度为 (1, 13, T)"""
    mfcc = zip(*python_speech_features.mfcc(audio_clip, sample_rate))
    audio_features = np.stack([np.array(i) for i in mfcc])
    audio_features = np.expand_dims(audio_features, axis=0)
    return audio_features


_fbank_generater_dict = {}
_fbank_normalizer = None


def get_fbank_generater(sample_rate: int) -> Fbank:
    if sample_rate in _fbank_generater_dict:
        return _fbank_generater_dict[sample_rate]

    n_mels = 80
    left_frames = 0
    right_frames = 0
    deltas = False

    fbank_generater = Fbank(
        sample_rate=sample_rate,
        n_mels=n_mels,
        left_frames=left_frames,
        right_frames=right_frames,
        deltas=deltas,
    )
    _fbank_generater_dict[sample_rate] = fbank_generater
    return fbank_generater


def generate_fbank(audio_clip: np.ndarray, sample_rate: int) -> torch.Tensor:
    """计算音频片段的梅尔频谱图，维度为 (1, T, 80)"""
    fbank_generater = get_fbank_generater(sample_rate)
    audio_tensor = torch.from_numpy(audio_clip).float()
    audio_tensor = torch.unsqueeze(audio_tensor, dim=0)
    fbank = fbank_generater(audio_tensor)
    return fbank


def get_fbank_normalizer() -> InputNormalization:
    global _fbank_normalizer
    if _fbank_normalizer is None:
        _fbank_normalizer = InputNormalization(norm_type="sentence", std_norm=False)
    return _fbank_normalizer


def normalize_fbank(
    fbank: torch.Tensor, len_ratio: torch.Tensor = torch.FloatTensor([1.0])
) -> torch.Tensor:
    """对 fbank 进行归一化
    :param fbank: (1, T, 80)
    :param len_ratio: (1,)
    """
    normalizer = get_fbank_normalizer()
    return normalizer(fbank, len_ratio)
