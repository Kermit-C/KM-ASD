#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线处理器
@Author: chenkeming
@Date: 2024-02-10 15:29:03
"""

from config import event_bus
from event_bus.event_bus_publisher import EventBusPublisher
from event_bus.event_message import EventMessage, EventMessageBody


class BaseEventBusProcessor:
    """事件总线处理器基类"""

    def __init__(self, processor_name: str):
        self.processor_name: str = processor_name
        self.processor_properties: dict = event_bus["processors"][self.__class__][
            "properties"
        ]

    def process(self, event_message_body: EventMessageBody):
        """处理消息，需要重写"""
        raise NotImplementedError("process method must be implemented")

    def publish_next(self, topic: str, messageBody: EventMessageBody):
        """发布下一个消息"""
        message = self._last_message.copy()
        message.body = messageBody
        self._get_publisher().publish(topic, message)

    def result(self, messageBody: EventMessageBody):
        """输出消息"""
        message = self._last_message.copy()
        message.body = messageBody
        self._last_message.result_consumer(message)

    def _handler(self, event_message: EventMessage):
        self._last_message = event_message
        return self.process(event_message.body)

    def _get_publisher(self) -> EventBusPublisher:
        """获取发布者"""
        if self._publisher is None:
            raise Exception("publisher is not set")
        return self._publisher

    def _set_publisher(self, publisher: EventBusPublisher):
        """设置发布者"""
        self._publisher = publisher
