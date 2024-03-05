#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import math
import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from active_speaker_detection.utils.augmentations import video_temporal_crop

from .data_store import DataStore


class VoiceFaceDataset(Dataset):

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

    def __len__(self):
        # 训练的话用实体列表
        return len(self.store.entity_list)

    def __getitem__(self, index):
        video_id, entity_id = self.store.entity_list[index]
        target_entity_metadata = self.store.entity_data[video_id][entity_id]

        # 找到有说话的索引列表
        target_entity_metadata_speak_idx_list = [
            idx if metadata[-1] == 1 else -1
            for idx, metadata in enumerate(target_entity_metadata)
        ]
        target_entity_metadata_speak_idx_list = list(
            filter(lambda x: x != -1, target_entity_metadata_speak_idx_list)
        )

        # 选择一个有说话片段中间时间戳的索引
        if len(target_entity_metadata_speak_idx_list) > 0:
            target_entity_metadata_speak_start_idx = (
                target_entity_metadata_speak_idx_list[0]
            )
            target_entity_metadata_speak_end_idx = (
                target_entity_metadata_speak_idx_list[0]
            )
            for i, idx in enumerate(target_entity_metadata_speak_idx_list):
                if idx - target_entity_metadata_speak_start_idx == i:
                    # 如果是连续的才记录
                    target_entity_metadata_speak_end_idx = idx
                else:
                    break
            # 从中后部分随机选
            center_index = random.randint(
                (
                    target_entity_metadata_speak_start_idx
                    + target_entity_metadata_speak_end_idx
                )
                // 2,
                target_entity_metadata_speak_end_idx,
            )
        else:
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
        )
        # 获取音频特征和标签
        audio_data, audio_fbank, target_a, entity_a = self.store.get_audio_data(
            video_id,
            entity_id,
            center_index,
            self.half_clip_length,
        )
        audio_data = torch.from_numpy(audio_data)
        audio_fbank = torch.from_numpy(audio_fbank)

        audio_index = (
            self.store.entity_list.index((video_id, entity_a)) if entity_a != "" else -1
        )  # 音频实体序号，-1 代表没有，则是环境音

        return (
            torch.stack(video_data[0], dim=1),  # video_data: (3, T, H, W)
            audio_data,
            audio_fbank,
            audio_index,  # 音频实体序号，-1 代表没有，则是环境音
            index,  # 视频实体序号
            timestamp,
        )
