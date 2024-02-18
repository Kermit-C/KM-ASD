#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 说话人验证处理器
@Author: Kermit
@Date: 2024-02-18 15:58:15
"""


from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.speaker_verificate_message_body import (
    SpeakerVerificateMessageBody,
)


class SpeakerVerificateProcessor(BaseEventBusProcessor):
    """说话人验证处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: SpeakerVerificateMessageBody):
        pass
