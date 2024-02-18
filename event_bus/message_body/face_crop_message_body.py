#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:10:50
"""

from typing import Any

import numpy as np

from event_bus.event_message import EventMessageBody


class FaceCropMessageBody(EventMessageBody):

    def __init__(
        self,
        frame_count: int,
        frame_timestamp: int,
        frame: np.ndarray,
        face_dets: list[dict[str, Any]],
    ):
        # 帧数，代表第几帧
        self.frame_count = frame_count
        # 帧时间戳
        self.frame_timestamp = frame_timestamp
        # 帧图像
        self.frame = frame
        # 人脸检测结果
        self.face_dets = face_dets
