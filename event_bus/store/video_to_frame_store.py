#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 视频转图片帧存储
@Author: Kermit
@Date: 2024-02-19 14:54:10
"""

from typing import Callable, Optional

import numpy as np

from store.store import Store


class VideoToFrameStore:

    def __init__(
        self, store_creater: Callable[[bool, int], Store], max_request_count: int = 1000
    ):
        self.store_creater = store_creater
        self.frame_store_of_request = store_creater(True, max_request_count)
        self.info_store_of_request = store_creater(True, max_request_count * 2)

    def save_frame(
        self,
        request_id: str,
        frame_count: int,
        frame_timestamp: float,
        frame: np.ndarray,
    ):
        if not self.frame_store_of_request.has(request_id):
            self.frame_store_of_request.put(
                request_id, {"frames": [], "frames_ts_to_cnt_dict": {}}
            )
        request_store = self.frame_store_of_request.get(request_id)
        while len(request_store["frames"]) <= frame_count - 1:
            # 补充空帧
            request_store["frames"].append(None)
        request_store["frames"][frame_count - 1] = frame
        request_store["frames_ts_to_cnt_dict"][frame_timestamp] = frame_count
        # TODO: 实现实时检测后需要考虑删除过期帧

    def save_info(
        self,
        request_id: str,
        video_fps: float,
        video_frame_count: int,
    ):
        self.info_store_of_request.put(
            request_id, {"video_fps": video_fps, "video_frame_count": video_frame_count}
        )

    def get_frames(self, request_id: str) -> Optional[list[np.ndarray]]:
        if not self.frame_store_of_request.has(request_id):
            return None
        return self.frame_store_of_request.get(request_id)["frames"]

    def get_frame_from_timestamp(
        self, request_id: str, timestamp: float
    ) -> Optional[np.ndarray]:
        if not self.frame_store_of_request.has(request_id):
            return None
        frame_count = self.frame_store_of_request.get(request_id)[
            "frames_ts_to_cnt_dict"
        ][timestamp]
        return self.frame_store_of_request.get(request_id)["frames"][frame_count - 1]

    def get_info(self, request_id: str) -> dict:
        if not self.info_store_of_request.has(request_id):
            return {}
        return self.info_store_of_request.get(request_id)
