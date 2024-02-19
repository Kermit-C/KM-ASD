#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:12:33
"""

from typing import Optional

import numpy as np

from ..event_message import EventMessageBody


class ReduceMessageBody(EventMessageBody):

    def __init__(
        self,
        type: str,  # ASD, FACE_RECOGNIZE, SPEAKER_VERIFICATE
        # ASD & FACE_RECOGNIZE COMMON
        frame_count: Optional[int] = None,
        frame_timestamp: Optional[int] = None,
        frame_face_idx: Optional[int] = None,  # 人脸索引，一个视频帧中可能有多个人脸
        # ASD
        frame_face_count: Optional[int] = None,  # 人脸数量
        frame_face_bbox: Optional[tuple[int, int, int, int]] = None,  # [x1, y1, x2, y2]
        frame_asd_status: Optional[int] = None,  # 0:未说话，1:说话
        # FACE_RECOGNIZE
        frame_face_label: Optional[str] = None,
        # SPEAKER_VERIFICATE
        audio_sample_rate: Optional[int] = None,
        audio_frame_length: Optional[int] = None,
        audio_frame_step: Optional[int] = None,
        audio_frame_count: Optional[int] = None,
        audio_frame_timestamp: Optional[int] = None,
        frame_voice_label: Optional[str] = None,
    ):
        self.type = type
        # ASD & FACE_RECOGNIZE COMMON
        self.frame_count = frame_count
        self.frame_timestamp = frame_timestamp
        self.frame_face_idx = frame_face_idx
        # ASD
        self.frame_face_count = frame_face_count
        self.frame_face_bbox = frame_face_bbox
        self.frame_asd_status = frame_asd_status
        # FACE_RECOGNIZE
        self.frame_face_label = frame_face_label
        # SPEAKER_VERIFICATE
        self.audio_sample_rate = audio_sample_rate
        self.audio_frame_length = audio_frame_length
        self.audio_frame_step = audio_frame_step
        self.audio_frame_count = audio_frame_count
        self.audio_frame_timestamp = audio_frame_timestamp
        self.frame_voice_label = frame_voice_label
