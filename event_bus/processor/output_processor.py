#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 结果输出
@Author: Kermit
@Date: 2024-02-18 15:55:11
"""


from event_bus.event_bus_processor import BaseEventBusProcessor


class OutputProcessor(BaseEventBusProcessor):
    """结果输出处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body):
        pass
