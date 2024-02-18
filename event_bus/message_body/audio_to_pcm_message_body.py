#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 16:31:41
"""


from event_bus.event_message import EventMessageBody


class AudioToPcmMessageBody(EventMessageBody):

    def __init__(self, audio_path: str):
        self.audio_path = audio_path
