#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 音频转 PCM 帧存储
@Author: Kermit
@Date: 2024-02-19 14:54:10
"""

from typing import Callable, Optional

import numpy as np
import torch

from store.store import Store


class AudioToPcmStore:

    def __init__(
        self, store_creater: Callable[[bool, int], Store], max_request_count: int = 1000
    ):
        self.store_creater = store_creater
        self.store_of_request = store_creater(True, max_request_count)

    def save_frame(
        self,
        request_id: str,
        audio_pcm: torch.Tensor,
        audio_sample_rate: int,
        audio_frame_length: int,
        audio_frame_step: int,
        audio_frame_count: int,
        audio_frame_timestamp: int,
    ):
        if not self.store_of_request.has(request_id):
            self.store_of_request.put(
                request_id,
                {
                    "frames": [],
                    "frames_ts_to_cnt_dict": {},
                    "audio_sample_rate": audio_sample_rate,
                    "audio_frame_length": audio_frame_length,
                    "audio_frame_step": audio_frame_step,
                },
            )
        request_store = self.store_of_request.get(request_id)
        while len(request_store["frames"]) <= audio_frame_count - 1:
            # 补充空帧
            request_store["frames"].append(None)
        request_store["frames"][audio_frame_count - 1] = audio_pcm
        request_store["frames_ts_to_cnt_dict"][
            audio_frame_timestamp
        ] = audio_frame_count

    def get_frames(self, request_id: str) -> Optional[list[np.ndarray]]:
        if not self.store_of_request.has(request_id):
            return None
        return self.store_of_request.get(request_id)["frames"]

    def get_frame_from_timestamp(
        self, request_id: str, timestamp: float
    ) -> Optional[np.ndarray]:
        if not self.store_of_request.has(request_id):
            return None
        frame_count = self.store_of_request.get(request_id)["frames_ts_to_cnt_dict"][
            timestamp
        ]
        return self.store_of_request.get(request_id)["frames"][frame_count - 1]
