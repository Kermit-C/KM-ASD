#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:13:07
"""

from typing import Optional

import torch

from event_bus.event_message import EventMessageBody


class SpeakerVerificateMessageBody(EventMessageBody):

    def __init__(
        self,
        audio_pcm: torch.Tensor,
        audio_sample_rate: int,
        audio_frame_length: int,
        audio_frame_step: int,
        audio_frame_count: int,
        audio_frame_timestamp: int,
    ):
        self.audio_pcm = audio_pcm
        self.audio_sample_rate = audio_sample_rate
        self.audio_frame_length = audio_frame_length
        self.audio_frame_step = audio_frame_step
        self.audio_frame_count = audio_frame_count
        self.audio_frame_timestamp = audio_frame_timestamp
