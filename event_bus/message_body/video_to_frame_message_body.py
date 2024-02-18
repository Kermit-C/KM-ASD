#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:57:22
"""

from event_bus.event_message import EventMessageBody


class VideoToFrameMessageBody(EventMessageBody):

    def __init__(self, video_path: str):
        # TODO
        # 视频路径
        self.video_path = video_path
