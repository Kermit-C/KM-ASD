#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:11:29
"""

import numpy as np

from event_bus.event_message import EventMessageBody


class FaceDetectMessageBody(EventMessageBody):

    def __init__(self, frame_count: int, frame_timestamp: int, frame: np.ndarray):
        # 帧数，代表第几帧
        self.frame_count = frame_count
        # 帧时间戳
        self.frame_timestamp = frame_timestamp
        # 帧图像
        self.frame = frame
