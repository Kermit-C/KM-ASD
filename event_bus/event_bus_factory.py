#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线工厂
@Author: chenkeming
@Date: 2024-02-10 15:20:03
"""

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.event_bus_publisher import EventBusPublisher

# 全局发布者和处理器
publisher_map: map[str, EventBusPublisher] = map()
processor_map: map[str, BaseEventBusProcessor] = map()


def get_publisher(publisher_name: str) -> EventBusPublisher:
    """获取发布者"""
    if publisher_name not in publisher_map:
        publisher_map[publisher_name] = EventBusPublisher(publisher_name)
    return publisher_map[publisher_name]


def create_processor(processor: BaseEventBusProcessor, publisher_name: str, topic: str):
    """创建处理器"""
    if processor.processor_name in processor_map:
        raise Exception(f"processor {processor.processor_name} already exists")
    publisher = get_publisher(publisher_name)
    publisher._subscribe(processor, topic)
    processor._set_publisher(publisher)
    processor_map[processor.processor_name] = processor
    return processor
