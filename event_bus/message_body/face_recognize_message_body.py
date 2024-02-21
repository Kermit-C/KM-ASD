#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:12:05
"""

import numpy as np

from event_bus.event_message import EventMessageBody


class FaceRecognizeMessageBody(EventMessageBody):

    def __init__(
        self,
        frame_count: int,
        frame_timestamp: int,
        frame: np.ndarray,
        frame_face_idx: int,
        frame_face_bbox: tuple[int, int, int, int],
        face_lmks: np.ndarray,
    ):
        # 帧数，代表第几帧
        self.frame_count = frame_count
        # 帧时间戳
        self.frame_timestamp = frame_timestamp
        # 帧图像
        self.frame = frame
        # 人脸索引
        self.frame_face_idx = frame_face_idx
        # 人脸框 (x1, y1, x2, y2)
        self.frame_face_bbox = frame_face_bbox
        # 人脸关键点 (5, 2)，相对于人脸框的坐标
        self.face_lmks = face_lmks
