#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线处理器
@Author: chenkeming
@Date: 2024-02-10 15:29:03
"""

import threading

from config import event_bus

from .event_bus_excecutor import submit as submit_executor
from .event_message import EventMessage, EventMessageBody


class BaseEventBusProcessor:
    """事件总线处理器基类"""

    def __init__(self, processor_name: str):
        self.processor_name: str = processor_name
        self.processor_properties: dict = event_bus["processors"][
            self.__class__.__name__
        ]["properties"]
        self.processor_timeout: float = event_bus["processors"][
            self.__class__.__name__
        ]["timeout"]
        self.last_message: threading.local = threading.local()

    def process(self, event_message_body: EventMessageBody):
        """处理消息，需要重写"""
        raise NotImplementedError("process method must be implemented")

    def process_exception(
        self, event_message_body: EventMessageBody, exception: Exception
    ):
        """处理异常，需要重写
        如果不重写或方法里也抛出，则向整个 request 抛出异常"""
        raise exception

    def publish_next(
        self,
        topic: str,
        messageBody: EventMessageBody,
        is_async: bool = True,
        is_wait_async: bool = False,
        wait_async_timeout: float = 0.0,
    ):
        """发布下一个消息"""
        message = self.last_message.value.copy()
        message.body = messageBody
        message.is_async = is_async
        message.is_wait_async = is_wait_async
        message.wait_async_timeout = wait_async_timeout
        self._get_publisher().publish(topic, message)

    def result(self, messageBody: EventMessageBody):
        """输出消息"""
        message = self.last_message.value.copy()
        message.body = messageBody
        self.last_message.value.result_consumer(message)

    def result_exception(self, exception: Exception):
        """输出异常"""
        message = self.last_message.value.copy()
        message.body = exception
        self.last_message.value.result_consumer(message)

    def get_request_id(self) -> str:
        return self.last_message.value.request_id

    def get_request_timestamp(self) -> int:
        return self.last_message.value.timestamp

    def is_real_time(self) -> bool:
        return self.last_message.value.is_real_time

    def _handler(self, event_message: EventMessage):
        """处理消息"""
        try:
            self.last_message.value = event_message
            assert isinstance(event_message.body, EventMessageBody)
            return self.process(event_message.body)
        except Exception as e:
            try:
                assert isinstance(event_message.body, EventMessageBody)
                self.process_exception(event_message.body, e)
            except Exception as ee:
                self.result_exception(ee)

    def _listener(self, event_message: EventMessage):
        """监听消息"""
        if event_message.is_async:
            # 异步处理
            handle_future = submit_executor(self._handler, event_message)
            if event_message.is_wait_async:
                # 等待异步处理
                try:
                    handle_future.result(event_message.wait_async_timeout)
                except:
                    pass
        else:
            # 同步处理
            self._handler(event_message)

    def _get_publisher(self):
        """获取发布者"""
        if self._publisher is None:
            raise Exception("publisher is not set")
        return self._publisher

    def _set_publisher(self, publisher):
        """设置发布者"""
        self._publisher = publisher
