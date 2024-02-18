#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:10:02
"""

from typing import Optional

import numpy as np
import torch

from event_bus.event_message import EventMessageBody


class AsdMessageBody(EventMessageBody):

    def __init__(
        self,
        type: str,  # V or A
        # V 视频消息体
        frame_count: Optional[int] = None,
        frame_timestamp: Optional[int] = None,
        frame: Optional[np.ndarray] = None,
        # A 音频消息体
        audio_pcm: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = None,
        audio_frame_length: Optional[int] = None,
        audio_frame_step: Optional[int] = None,
        audio_frame_count: Optional[int] = None,
        audio_frame_timestamp: Optional[int] = None,
    ):
        self.type = type
        # V 视频消息体
        self.frame_count = frame_count
        self.frame_timestamp = frame_timestamp
        self.frame = frame

        # A 音频消息体
        self.audio_pcm = audio_pcm
        self.audio_sample_rate = audio_sample_rate
        self.audio_frame_length = audio_frame_length
        self.audio_frame_step = audio_frame_step
        self.audio_frame_count = audio_frame_count
        self.audio_frame_timestamp = audio_frame_timestamp
