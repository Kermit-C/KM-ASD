#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.io import wavfile

from active_speaker_detection.utils.audio_processing import (
    generate_fbank,
    generate_mel_spectrogram,
    normalize_fbank,
)


def _pil_loader(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def _cached_pil_loader(path, cache):
    if path in cache.keys():
        return cache[path]

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        cache[path] = rgb
        return rgb


def _fit_audio_clip(audio_clip: np.ndarray, sample_rate: int, video_clip_length):
    # 1 / 27 表示数据集里的每个 clip 是多少秒，数据集是 25fps，这里略小于一点
    target_audio_length = int((1.0 / 27.0) * sample_rate * video_clip_length)
    pad_required = int((target_audio_length - len(audio_clip)) / 2)
    if pad_required > 0:
        audio_clip = np.pad(
            audio_clip, pad_width=(pad_required, pad_required), mode="reflect"
        )
    if pad_required < 0:
        audio_clip = audio_clip[-1 * pad_required : pad_required]

    # TODO There is a +-1 offset here and I dont feel like cheking it
    return audio_clip[0 : target_audio_length - 1]


def load_v_clip_from_metadata(clip_meta_data, frames_source) -> List[Image.Image]:
    """从视频图片序列加载 Image"""
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    entity_id = clip_meta_data[0][0]

    selected_frames = [
        os.path.join(frames_source, entity_id.replace(":", "_"), ts + ".jpg")
        for ts in ts_sequence
    ]
    video_data = [_pil_loader(sf) for sf in selected_frames]
    return video_data


def load_v_clip_from_metadata_cache(
    clip_meta_data: List[Tuple[str, str, int]],
    frames_source: str,
    cache: dict,
    entity_cache: Optional[dict] = None,
) -> List[Image.Image]:
    """从视频图片序列加载 Image，使用缓存"""
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]
    entity_id = clip_meta_data[0][0]

    video_data = []
    for ts in ts_sequence:
        if entity_cache is not None and entity_id + "_" + ts in entity_cache.keys():
            video_data.append(entity_cache[entity_id + "_" + ts])
        else:
            video_data.append(
                _cached_pil_loader(
                    os.path.join(
                        frames_source, entity_id.replace(":", "_"), ts + ".jpg"
                    ),
                    cache,
                )
            )

    if entity_cache is not None:
        for img, meta in zip(video_data, clip_meta_data):
            ts = meta[1]
            entity_cache[entity_id + "_" + ts] = img

    return video_data


def load_a_clip_from_metadata(
    clip_meta_data: List[Tuple[str, str, int]],
    frames_source,
    audio_source,
    audio_offset: float,
    entity_cache: Optional[dict] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """从片段元数据中获得音频梅尔特征"""
    # 从片段元数据中获得时间戳序列
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    # 最小时间戳和最大时间戳
    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    # 实体ID
    entity_id = clip_meta_data[0][0]

    if entity_cache is not None and entity_id in entity_cache.keys():
        sample_rate, audio_data = entity_cache[entity_id]
    else:
        # 音频文件
        audio_file = os.path.join(audio_source, entity_id.replace(":", "_") + ".wav")
        # audio_data 是一个 numpy.ndarray，int16 pcm格式
        sample_rate, audio_data = wavfile.read(audio_file)

    if entity_cache is not None:
        entity_cache[entity_id] = (sample_rate, audio_data)

    # 通过时间戳和采样率计算音频起始和结束位置，位置是采样点
    audio_start = int((min_ts - audio_offset) * sample_rate)
    audio_end = int((max_ts - audio_offset) * sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(ts_sequence))
    audio_features = generate_mel_spectrogram(audio_clip, sample_rate)
    audio_fbank = generate_fbank(audio_clip, sample_rate)
    audio_fbank = normalize_fbank(audio_fbank, torch.FloatTensor([1.0]))

    return audio_features, audio_fbank
