#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸识别处理器
@Author: Kermit
@Date: 2024-02-18 15:51:15
"""

from event_bus.event_bus_processor import BaseEventBusProcessor


class FaceRecognizeProcessor(BaseEventBusProcessor):
    """人脸识别处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body):
        pass
