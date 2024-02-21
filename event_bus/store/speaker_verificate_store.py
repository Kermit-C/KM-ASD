#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-21 16:09:46
"""


from threading import Lock
from typing import Callable, Optional

import numpy as np
import torch

from store.store import Store


class SpeakerVerificateStore:
    def __init__(
        self,
        store_creater: Callable[[bool, int], Store],
        max_request_count: int = 1000,
        # 保留的最大帧数
        max_frame_count: int = 1000,
    ):
        self.store_creater = store_creater
        self.store_of_request = store_creater(True, max_request_count)
        self.max_frame_count = max_frame_count
        self.save_frame_lock = Lock()

    def save_frame(
        self,
        request_id: str,
        audio_frame_count: int,
        audio_frame_timestamp: int,
        audio_sample_rate: int,
        audio_frame_length: int,
        audio_frame_step: int,
        audio_frame: torch.Tensor,
    ):
        with self.save_frame_lock:
            if not self.store_of_request.has(request_id):
                self.store_of_request.put(
                    request_id,
                    {
                        "frames": [],
                        "audio_sample_rate": audio_sample_rate,
                        "audio_frame_length": audio_frame_length,
                        "audio_frame_step": audio_frame_step,
                    },
                )
            request_store = self.store_of_request.get(request_id)
            while len(request_store["frames"]) <= audio_frame_count - 1:
                # 补充空帧
                request_store["frames"].append(None)
            if request_store["frames"][audio_frame_count - 1] is None:
                request_store["frames"][audio_frame_count - 1] = {
                    "audio_frame_timestamp": audio_frame_timestamp,
                    "audio_frame": audio_frame,
                    "audio_frame_label": None,
                }
            # 保留的最大帧数
            if len(request_store["frames"]) > self.max_frame_count:
                request_store["frames"][: -self.max_frame_count] = [None] * (
                    len(request_store["frames"]) - self.max_frame_count
                )

    def save_frame_label(
        self,
        request_id: str,
        audio_frame_count: int,
        audio_frame_label: str,
    ):
        if not self.store_of_request.has(request_id):
            return
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frames"]) < audio_frame_count:
            return
        request_store["frames"][audio_frame_count - 1][
            "audio_frame_label"
        ] = audio_frame_label

    def get_frame(self, request_id: str, audio_frame_count: int):
        if not self.store_of_request.has(request_id):
            return None
        if len(self.store_of_request.get(request_id)["frames"]) < audio_frame_count:
            return None
        frame = self.store_of_request.get(request_id)["frames"][audio_frame_count - 1]
        return {
            "audio_frame_timestamp": frame["audio_frame_timestamp"],
            "audio_frame": frame["audio_frame"],
            "audio_frame_label": frame["audio_frame_label"],
        }

    def get_info(self, request_id: str):
        if not self.store_of_request.has(request_id):
            return None
        return {
            "audio_sample_rate": self.store_of_request.get(request_id)[
                "audio_sample_rate"
            ],
            "audio_frame_length": self.store_of_request.get(request_id)[
                "audio_frame_length"
            ],
            "audio_frame_step": self.store_of_request.get(request_id)[
                "audio_frame_step"
            ],
        }
