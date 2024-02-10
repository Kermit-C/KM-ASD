#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:57:22
"""

from event_bus.event_message import EventMessageBody


class VideoToImageMessageBody(EventMessageBody):
    def __init__(self, video_path: str, output_dir: str, frame_rate: int = 1):
        # TODO
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_rate = frame_rate
