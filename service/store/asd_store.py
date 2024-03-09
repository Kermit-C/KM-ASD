#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-22 10:49:23
"""

from threading import RLock
from typing import Callable, Optional

import numpy as np

from store.store import Store


class ActiveSpeakerDetectionStore:
    """说话人检测存储"""

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
        self.save_frame_lock = RLock()

    def save_frame(
        self,
        request_id: str,
        frame_count: int,
        face_frame: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        audio_frame: np.ndarray,
    ):
        with self.save_frame_lock:
            if not self.store_of_request.has(request_id):
                self.store_of_request.put(request_id, {"frames": []})
            request_store = self.store_of_request.get(request_id)
            while len(request_store["frames"]) <= frame_count - 1:
                # 补充空帧
                request_store["frames"].append(None)
            if request_store["frames"][frame_count - 1] is None:
                request_store["frames"][frame_count - 1] = {
                    "faces": [],
                    "audio_frame": audio_frame,
                }
            request_store["frames"][frame_count - 1]["faces"].append(
                {
                    "face_frame": face_frame,
                    "face_bbox": face_bbox,
                }
            )
            # 保留的最大帧数
            if len(request_store["frames"]) > self.max_frame_count:
                request_store["frames"][: -self.max_frame_count] = [None] * (
                    len(request_store["frames"]) - self.max_frame_count
                )

    def get_frame_faces(self, request_id: str, frame_count: int) -> list[dict]:
        if frame_count < 1:
            return []
        if not self.store_of_request.has(request_id):
            return []
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frames"]) < frame_count:
            return []
        if request_store["frames"][frame_count - 1] is None:
            return []
        return request_store["frames"][frame_count - 1]["faces"]

    def get_frame_audio(
        self, request_id: str, frame_count: int
    ) -> Optional[np.ndarray]:
        if frame_count < 1:
            return None
        if not self.store_of_request.has(request_id):
            return None
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frames"]) < frame_count:
            return None
        if request_store["frames"][frame_count - 1] is None:
            return None
        return request_store["frames"][frame_count - 1]["audio_frame"]
