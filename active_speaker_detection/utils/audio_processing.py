#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import numpy as np
import python_speech_features


def generate_mel_spectrogram(audio_clip: np.ndarray, sample_rate: int) -> np.ndarray:
    """计算音频片段的梅尔频谱图，维度为 (1, 13, T)"""
    mfcc = zip(*python_speech_features.mfcc(audio_clip, sample_rate))
    audio_features = np.stack([np.array(i) for i in mfcc])
    audio_features = np.expand_dims(audio_features, axis=0)
    return audio_features
