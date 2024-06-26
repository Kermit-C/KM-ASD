#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import math
import random
import sys
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from active_speaker_detection.utils.augmentations import video_temporal_crop

from .data_store import DataStore


class EncoderDataset(Dataset):

    def __init__(
        self,
        audio_root,
        video_root,
        data_store_train_cache,
        clip_length,  # 片段长度，短时序上下文片段的长度
        video_transform: Optional[transforms.Compose] = None,  # 视频转换方法
        do_video_augment=False,  # 是否视频增强
        crop_ratio=0.95,  # 视频裁剪比例
        norm_audio=False,  # 是否归一化音频
        eval=False,
    ):
        super().__init__()
        self.store = DataStore(audio_root, video_root, data_store_train_cache)

        # 后处理
        self.crop_ratio = crop_ratio
        self.video_transform = (
            video_transform if video_transform is not None else transforms.ToTensor()
        )
        self.do_video_augment = do_video_augment  # 是否视频增强

        # 图节点的配置
        self.norm_audio = norm_audio  # 是否归一化音频
        self.half_clip_length = math.floor(
            clip_length / 2
        )  # 每刻计算特征的帧数一半的长度

        self.eval = eval

        if self.eval:
            self.video_entity_cache = (
                {}
            )  # 以实体 + 时间戳为 key，主要用于依次计算特征的缓存
            self.audio_entity_cache = {}  # 以实体为 key，主要用于依次计算特征的缓存
        else:
            self.video_entity_cache = None
            self.audio_entity_cache = None

    def __len__(self):
        if self.eval:
            # 是 eval 就用全部
            return len(self.store.feature_list)
        else:
            # 训练的话用实体列表
            return len(self.store.entity_list)

    def __getitem__(self, index):
        if self.eval:
            video_id, entity_id, timestamp, entity_label = self.store.feature_list[
                index
            ]
            target_entity_metadata = self.store.entity_data[video_id][entity_id]
            center_index = self.store.search_ts_in_meta_data(
                target_entity_metadata, timestamp
            )
        else:
            video_id, entity_id = self.store.entity_list[index]
            target_entity_metadata = self.store.entity_data[video_id][entity_id]
            # 随机选择一个中间时间戳的索引
            # TODO: 这样没把所有的时间戳都用上，影响大吗？
            center_index = random.randint(0, len(target_entity_metadata) - 1)
            timestamp = target_entity_metadata[center_index][1]

        # 获取视频特征和标签
        cache = {}  # 以图片路径为 key 的缓存
        video_data, targets, entities, positions = self.store.get_video_data(
            video_id,
            entity_id,
            center_index,
            1,
            self.half_clip_length,
            self.video_transform,
            video_temporal_crop if self.do_video_augment else None,
            self.crop_ratio if self.do_video_augment else None,
            cache,
            self.video_entity_cache,
        )
        # 获取音频特征和标签
        audio_data, audio_fbank, target_a, entity_a = self.store.get_audio_data(
            video_id,
            entity_id,
            center_index,
            self.half_clip_length,
            self.audio_entity_cache,
        )
        audio_data = torch.from_numpy(audio_data)
        audio_fbank = torch.from_numpy(audio_fbank)
        # 一个 self.video_entity_cache 在 10KB 左右，即人脸切片视频帧大小？（不准确）
        # 以实体 + 时间戳为 key，主要用于依次计算特征的缓存
        # 一个 self.audio_entity_cache 在 150KB 左右，即音频 wav 文件大小（不准确）
        # 以实体为 key，主要用于依次计算特征的缓存
        # 限制 self.video_entity_cache 长度，最大 500，从旧的开始删除
        # 限制 self.audio_entity_cache 长度，最大 10，从旧的开始删除
        # 这个配置比例经过测试很均衡，单 worker 占用内存 500MB 左右
        while (
            self.video_entity_cache is not None and len(self.video_entity_cache) > 500
        ):
            self.video_entity_cache.pop(list(self.video_entity_cache.keys())[0])
        while self.audio_entity_cache is not None and len(self.audio_entity_cache) > 10:
            self.audio_entity_cache.pop(list(self.audio_entity_cache.keys())[0])

        video_index = (
            index
            if self.eval
            else self.store.entity_list.index((video_id, entities[0]))
        )
        audio_index = (
            self.store.entity_list.index((video_id, entity_a)) if entity_a != "" else -1
        )  # 音频实体序号，-1 代表没有，则是环境音

        return (
            torch.stack(video_data[0], dim=1),  # video_data: (3, T, H, W)
            audio_data,
            audio_fbank,
            targets[0],  # 视频标签
            target_a,  # 音频标签
            audio_index,  # 音频实体序号，-1 代表没有，则是环境音
            video_index,  # 视频实体序号
            entities[0],  # 视频实体
            timestamp,
        )
