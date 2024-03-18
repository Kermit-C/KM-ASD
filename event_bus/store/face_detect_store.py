#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-19 15:50:28
"""

from asyncio import Lock
from typing import Any, Callable, Optional

import numpy as np

from store.store import Store


class FaceDetectStore:
    def __init__(
        self,
        store_creater: Callable[[bool, int], Store],
        max_request_count: int = 1000,
        # 保留的最大帧数
        max_frame_count: int = 50,
    ):
        self.store_creater = store_creater
        self.store_of_request = store_creater(True, max_request_count)
        self.max_frame_count = max_frame_count
        self.save_face_lock = Lock()

    async def save_face(
        self,
        request_id: str,
        frame_count: int,
        frame_timestamp: int,
        face_dets,
    ):
        async with self.save_face_lock:
            if not self.store_of_request.has(request_id):
                self.store_of_request.put(request_id, {"frames": []})
            request_store = self.store_of_request.get(request_id)
            while len(request_store["frames"]) <= frame_count - 1:
                # 补充空帧
                request_store["frames"].append(None)
            if request_store["frames"][frame_count - 1] is None:
                request_store["frames"][frame_count - 1] = {
                    "frame_timestamp": frame_timestamp,
                    "face_dets": None,
                }
            request_store["frames"][frame_count - 1]["face_dets"] = face_dets
            # 保留的最大帧数
            if len(request_store["frames"]) > self.max_frame_count:
                request_store["frames"][: -self.max_frame_count] = [None] * (
                    len(request_store["frames"]) - self.max_frame_count
                )

    def get_faces(self, request_id: str, frame_count: int) -> Optional[Any]:
        if frame_count < 1:
            return None
        if not self.store_of_request.has(request_id):
            return None
        if len(self.store_of_request.get(request_id)["frames"]) < frame_count:
            return None
        return self.store_of_request.get(request_id)["frames"][frame_count - 1][
            "face_dets"
        ]

    def get_frame_timestamp(self, request_id: str, frame_count: int) -> Optional[int]:
        if frame_count < 1:
            return None
        if not self.store_of_request.has(request_id):
            return None
        if len(self.store_of_request.get(request_id)["frames"]) < frame_count:
            return None
        return self.store_of_request.get(request_id)["frames"][frame_count - 1][
            "frame_timestamp"
        ]
