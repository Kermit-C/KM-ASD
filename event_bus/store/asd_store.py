#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-22 17:37:10
"""

from threading import Lock
from typing import Callable, Optional

import numpy as np
import torch

from store.store import Store


class ActiveSpeakerDetectionStore:

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
        frame_count: int,
        frame_timestamp: int,
        face_frame: np.ndarray,
        face_idx: int,  # 人脸索引，一个视频帧中可能有多个人脸
        face_bbox: tuple[int, int, int, int],
        frame_face_count: int,  # 人脸数量
        frame_height: int,  # 视频帧高度
        frame_width: int,  # 视频帧宽度
    ):
        with self.save_frame_lock:
            if not self.store_of_request.has(request_id):
                self.store_of_request.put(
                    request_id, {"frames": [], "frame_asded_count": 0}
                )
            request_store = self.store_of_request.get(request_id)
            while len(request_store["frames"]) <= frame_count - 1:
                # 补充空帧
                request_store["frames"].append(None)
            if request_store["frames"][frame_count - 1] is None:
                request_store["frames"][frame_count - 1] = {
                    "frame_timestamp": frame_timestamp,
                    "frame_face_count": frame_face_count,
                    "frame_height": frame_height,
                    "frame_width": frame_width,
                    "faces": [],
                    "audio_frame": None,
                    "is_asded": False,
                }
            else:
                request_store["frames"][frame_count - 1][
                    "frame_timestamp"
                ] = frame_timestamp
                request_store["frames"][frame_count - 1][
                    "frame_face_count"
                ] = frame_face_count
                request_store["frames"][frame_count - 1]["frame_height"] = frame_height
                request_store["frames"][frame_count - 1]["frame_width"] = frame_width
            request_store["frames"][frame_count - 1]["faces"].append(
                {
                    "face_idx": face_idx,
                    "face_frame": face_frame,
                    "face_bbox": face_bbox,
                }
            )
            # 保留的最大帧数
            if len(request_store["frames"]) > self.max_frame_count:
                request_store["frames"][: -self.max_frame_count] = [None] * (
                    len(request_store["frames"]) - self.max_frame_count
                )

    def save_audio_frame(
        self,
        request_id: str,
        frame_count: int,
        frame_timestamp: int,
        audio_frame: np.ndarray,
    ):
        with self.save_frame_lock:
            if not self.store_of_request.has(request_id):
                self.store_of_request.put(
                    request_id, {"frames": [], "frame_asded_count": 0}
                )
            request_store = self.store_of_request.get(request_id)
            while len(request_store["frames"]) <= frame_count - 1:
                # 补充空帧
                request_store["frames"].append(None)
            if request_store["frames"][frame_count - 1] is None:
                request_store["frames"][frame_count - 1] = {
                    "frame_timestamp": frame_timestamp,
                    "frame_face_count": -1,
                    "frame_height": -1,
                    "frame_width": -1,
                    "faces": [],
                    "audio_frame": audio_frame,
                    "is_asded": False,
                }
            else:
                request_store["frames"][frame_count - 1]["audio_frame"] = audio_frame

    def is_frame_completed(self, request_id: str, frame_count: int) -> bool:
        if frame_count < 1:
            return False
        if not self.store_of_request.has(request_id):
            return False
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frames"]) < frame_count:
            return False
        return (
            request_store["frames"][frame_count - 1] is not None
            and len(request_store["frames"][frame_count - 1]["faces"])
            == request_store["frames"][frame_count - 1]["frame_face_count"]
            and request_store["frames"][frame_count - 1]["audio_frame"] is not None
        )

    def is_frame_asded(self, request_id: str, frame_count: int) -> bool:
        if frame_count < 1:
            return False
        if not self.store_of_request.has(request_id):
            return False
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frames"]) < frame_count:
            return False
        return (
            request_store["frames"] is not None
            and request_store["frames"][frame_count - 1]["is_asded"]
        )

    def is_frame_before_all_asded(self, request_id: str, frame_count: int) -> bool:
        if frame_count < 1:
            return False
        if not self.store_of_request.has(request_id):
            return False
        request_store = self.store_of_request.get(request_id)
        return request_store["frame_asded_count"] >= frame_count - 1

    def get_frame_after_all_completed(
        self, request_id: str, frame_count: int
    ) -> list[int]:
        if frame_count < 1:
            return []
        if not self.store_of_request.has(request_id):
            return []
        request_store = self.store_of_request.get(request_id)
        return [
            i
            for i in range(frame_count + 1, len(request_store["frames"]) + 1)
            if self.is_frame_completed(request_id, i)
        ]

    def set_frame_asded(self, request_id: str, frame_count: int):
        if frame_count < 1:
            return
        if not self.store_of_request.has(request_id):
            return
        request_store = self.store_of_request.get(request_id)
        request_store["frame_asded_count"] = frame_count
        if len(request_store["frames"]) < frame_count:
            return
        request_store["frames"][frame_count - 1]["is_asded"] = True

    def get_frame_info(self, request_id: str, frame_count: int) -> Optional[dict]:
        if frame_count < 1:
            return None
        if not self.store_of_request.has(request_id):
            return None
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frames"]) < frame_count:
            return None
        return request_store["frames"][frame_count - 1]

    def get_frame_faces(self, request_id: str, frame_count: int) -> list[dict]:
        if frame_count < 1:
            return []
        if not self.store_of_request.has(request_id):
            return []
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frames"]) < frame_count:
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
        return request_store["frames"][frame_count - 1]["audio_frame"]
