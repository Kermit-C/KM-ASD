#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 视频流提取存储
@Author: Kermit
@Date: 2024-02-16 21:51:06
"""


from typing import Callable

from .store import Store


class StreamExtractStore:
    def __init__(self, store_creater: Callable[[bool, int], Store]):
        self.store_creater = store_creater
        self.store = store_creater(True, 1000)

    def save_clip(self, request_id: str, ts: float, video_clip, audio_clip):
        pass
