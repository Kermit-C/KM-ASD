#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 说话人检测处理器
@Author: Kermit
@Date: 2024-02-18 15:53:31
"""


from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.asd_message_body import AsdMessageBody


class AsdProcessor(BaseEventBusProcessor):
    """说话人检测处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: AsdMessageBody):
        pass
