#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线处理器
@Author: chenkeming
@Date: 2024-02-10 15:29:03
"""

import asyncio
import contextvars
import threading
import time
from concurrent.futures import Future
from typing import Coroutine, Optional

from config import event_bus
from manager.metric_manager import create_collector
from utils.logger_util import eb_logger

from .event_bus_excecutor import check_running_loop, get_event_loop
from .event_bus_excecutor import submit as submit_executor
from .event_message import EventMessage, EventMessageBody


class BaseEventBusProcessor:
    """事件总线处理器基类"""

    def __init__(self, processor_name: str, is_async: bool = False):
        self.processor_name: str = processor_name
        # 本处理器是否异步实现（使用事件循环）
        self.is_async = is_async
        self.processor_properties: dict = event_bus["processors"][
            self.__class__.__name__
        ]["properties"]
        self.processor_timeout: float = event_bus["processors"][
            self.__class__.__name__
        ]["timeout"]

        self.last_message: threading.local = threading.local()
        self.last_message_async: contextvars.ContextVar = contextvars.ContextVar(
            "last_message_async"
        )

        self.metric_collector_of_duration = create_collector(
            f"eventbus_processor_{self.processor_name}_duration"
        )
        # TODO: 其他指标

    def process(self, event_message_body: EventMessageBody):
        """处理消息，需要重写"""
        pass

    def process_async(self, event_message_body: EventMessageBody) -> Coroutine:  # type: ignore
        """处理消息，需要重写，异步处理，与 process 方法二选一"""
        pass

    def process_exception(
        self, event_message_body: EventMessageBody, exception: Exception
    ):
        """处理异常，需要重写
        如果不重写或方法里也抛出，则向整个 request 抛出异常"""
        raise exception

    def process_exception_async(self, event_message_body: EventMessageBody, exception: Exception) -> Coroutine:  # type: ignore
        """处理异常，需要重写，异步处理
        如果不重写或方法里也抛出，则向整个 request 抛出异常，与 process_exception 方法二选一"""
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
        if self.is_async:
            message = self.last_message_async.get().copy()
        else:
            message = self.last_message.value.copy()
        message.body = messageBody
        message.is_async = is_async
        message.is_wait_async = is_wait_async
        message.wait_async_timeout = wait_async_timeout
        self._get_publisher().publish(topic, message)

    def result(self, messageBody: EventMessageBody):
        """输出消息"""
        if self.is_async:
            message = self.last_message_async.get().copy()
        else:
            message = self.last_message.value.copy()
        message.body = messageBody
        if self.is_async:
            self.last_message_async.get().result_consumer(message)
        else:
            self.last_message.value.result_consumer(message)

    def result_exception(self, exception: Exception):
        """输出异常"""
        if self.is_async:
            message = self.last_message_async.get().copy()
        else:
            message = self.last_message.value.copy()
        message.body = exception
        if self.is_async:
            self.last_message_async.get().result_consumer(message)
        else:
            self.last_message.value.result_consumer(message)

    def get_request_id(self) -> str:
        if self.is_async:
            return self.last_message_async.get().request_id
        else:
            return self.last_message.value.request_id

    def get_request_timestamp(self) -> int:
        if self.is_async:
            return self.last_message_async.get().timestamp
        else:
            return self.last_message.value.timestamp

    def is_real_time(self) -> bool:
        if self.is_async:
            return self.last_message_async.get().is_real_time
        else:
            return self.last_message.value.is_real_time

    def _handler(self, event_message: EventMessage):
        """处理消息"""
        start_time = time.time()
        eb_logger.debug(f"processor {self.processor_name} start")
        try:
            assert isinstance(event_message.body, EventMessageBody)
            self.last_message.value = event_message
            return self.process(event_message.body)
        except Exception as e:
            try:
                eb_logger.debug(f"processor {self.processor_name} exception: {str(e)}")
                assert isinstance(event_message.body, EventMessageBody)
                self.process_exception(event_message.body, e)
            except Exception as ee:
                eb_logger.error(
                    f"processor {self.processor_name} process_exception exception", ee
                )
                self.result_exception(ee)
        finally:
            eb_logger.debug(
                f"processor {self.processor_name} finished, cost {int((time.time() - start_time) * 1000)} ms"
            )
            self.metric_collector_of_duration.collect(time.time() - start_time)

    async def _handler_async(self, event_message: EventMessage):
        """处理消息，异步处理"""
        start_time = time.time()
        eb_logger.debug(f"processor {self.processor_name} start")
        try:
            assert isinstance(event_message.body, EventMessageBody)
            self.last_message_async.set(event_message)
            return await self.process_async(event_message.body)
        except Exception as e:
            try:
                eb_logger.debug(f"processor {self.processor_name} exception: {str(e)}")
                assert isinstance(event_message.body, EventMessageBody)
                await self.process_exception_async(event_message.body, e)
            except Exception as ee:
                eb_logger.error(
                    f"processor {self.processor_name} process_exception exception",
                    ee,
                )
                self.result_exception(ee)
        finally:
            eb_logger.debug(
                f"processor {self.processor_name} finished, cost {int((time.time() - start_time) * 1000)} ms"
            )
            self.metric_collector_of_duration.collect(time.time() - start_time)

    def _listener(self, event_message: EventMessage):
        """监听消息"""
        # 判断来源的处理器是否事件循环
        if check_running_loop():
            self._listener_async(event_message)
        else:
            self._listener_sync(event_message)

    def _listener_sync(self, event_message: EventMessage):
        """监听消息，同步处理（来源的处理器非事件循环）"""
        try:
            if event_message.is_async:
                # 异步任务
                if not self.is_async:
                    # 使用同步实现的处理，加入线程池
                    handle_future = submit_executor(self._handler, event_message)
                else:
                    # 异步实现的处理
                    handle_future = asyncio.run_coroutine_threadsafe(
                        self._handler_async(event_message), get_event_loop()
                    )
                if event_message.is_wait_async:
                    # 等待异步任务
                    handle_future.result(event_message.wait_async_timeout)
            else:
                # 同步处理
                if not self.is_async:
                    # 使用同步实现的处理
                    self._handler(event_message)
                else:
                    # 异步实现的处理，异步转同步
                    handle_future = asyncio.run_coroutine_threadsafe(
                        self._handler_async(event_message), get_event_loop()
                    )
                    handle_future.result()
        except:
            pass

    def _listener_async(self, event_message: EventMessage):
        """监听消息，异步处理（来源的处理器是事件循环）"""
        # 来自事件循环的处理器，全部异步处理
        event_message.is_async = True
        event_message.is_wait_async = False
        submit_executor(lambda: self._listener_sync(event_message))

    def _get_publisher(self):
        """获取发布者"""
        if self._publisher is None:
            raise Exception("publisher is not set")
        return self._publisher

    def _set_publisher(self, publisher):
        """设置发布者"""
        self._publisher = publisher
