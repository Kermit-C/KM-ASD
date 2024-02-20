#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线发布者
@Author: chenkeming
@Date: 2024-02-10 15:21:27
"""

from pubsub import pub

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.event_message import EventMessage


class EventBusPublisher:
    """事件总线发布者"""

    processor_list: list[tuple[str, BaseEventBusProcessor]] = []

    def __init__(self, publisher_name: str):
        self.publisher = pub.Publisher()
        self.publisher_name = publisher_name

    def publish(self, topic: str, message: EventMessage):
        """发布消息"""
        self.publisher.sendMessage(topic, event_message=message)

    def publish_batch(self, topic: str, messages: list[EventMessage]):
        """批量发布消息"""
        for message in messages:
            self.publish(topic, message)

    def _subscribe(self, processor: BaseEventBusProcessor, topic: str):
        """订阅 processor"""
        self.publisher.subscribe(processor._listener, topic)
        self.processor_list.append((topic, processor))
