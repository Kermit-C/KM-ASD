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
        face_lmks: np.ndarray,
    ):
        # 帧数，代表第几帧
        self.frame_count = frame_count
        # 帧时间戳
        self.frame_timestamp = frame_timestamp
        # 帧图像
        self.frame = frame
        # 人脸关键点 (5, 2)
        self.face_lmks = face_lmks
