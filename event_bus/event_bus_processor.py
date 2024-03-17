#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线处理器
@Author: chenkeming
@Date: 2024-02-10 15:29:03
"""

import asyncio
import threading
import time
from typing import Coroutine, Optional

from config import event_bus
from utils.logger_util import eb_logger

from .event_bus_excecutor import get_event_loop, get_executor
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

    def _handler(self, event_message: EventMessage) -> None:
        """处理消息"""
        start_time = time.time()
        eb_logger.debug(f"processor {self.processor_name} start")
        try:
            self.last_message.value = event_message
            assert isinstance(event_message.body, EventMessageBody)
            self.process(event_message.body)
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

    def _handler_async(self, event_message: EventMessage) -> Optional[asyncio.Future]:
        """处理消息，异步处理"""
        start_time = time.time()
        eb_logger.debug(f"processor {self.processor_name} start")

        self.last_message.value = event_message
        assert isinstance(event_message.body, EventMessageBody)
        process_coroutine: Coroutine = self.process_async(event_message.body)
        if not asyncio.iscoroutine(process_coroutine):
            return None

        def _handler_async_callback(future: asyncio.Future):
            try:
                future.result()
            except Exception as e:
                try:
                    eb_logger.debug(
                        f"processor {self.processor_name} exception: {str(e)}"
                    )
                    assert isinstance(event_message.body, EventMessageBody)
                    self.process_exception(event_message.body, e)
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

        process_future: asyncio.Future = get_event_loop().run_in_executor(
            get_executor(), get_event_loop().create_task, process_coroutine
        )
        process_future.add_done_callback(_handler_async_callback)
        return process_future

    def _listener(self, event_message: EventMessage):
        """监听消息"""
        if event_message.is_async:
            # 异步任务
            # 先尝试异步实现的处理
            handle_asyncio_future = self._handler_async(event_message)
            if handle_asyncio_future is None:
                # 使用同步实现的处理，加入线程池
                handle_future = submit_executor(self._handler, event_message)
                if event_message.is_wait_async:
                    # 等待异步任务
                    try:
                        handle_future.result(event_message.wait_async_timeout)
                    except:
                        pass
            elif event_message.is_wait_async:
                # 等待异步任务
                begin_time = time.time()
                while True:
                    if handle_asyncio_future.done():
                        handle_asyncio_future.result()
                        break
                    if time.time() - begin_time > event_message.wait_async_timeout:
                        break
                    time.sleep(0.1)
        else:
            # 同步处理
            # 先尝试异步实现的处理
            handle_asyncio_future = self._handler_async(event_message)
            if handle_asyncio_future is None:
                # 使用同步实现的处理
                self._handler(event_message)
            else:
                # 异步转同步
                while True:
                    if handle_asyncio_future.done():
                        handle_asyncio_future.result()
                        break
                    time.sleep(0.1)

    def _get_publisher(self):
        """获取发布者"""
        if self._publisher is None:
            raise Exception("publisher is not set")
        return self._publisher

    def _set_publisher(self, publisher):
        """设置发布者"""
        self._publisher = publisher
