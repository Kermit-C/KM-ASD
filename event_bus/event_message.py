#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线消息
@Author: chenkeming
@Date: 2024-02-09 17:16:48
"""

import time
from typing import Callable, Union


class EventMessageBody:
    pass


class EventMessage:

    def __init__(
        self,
        request_id: str,
        result_consumer: Callable[["EventMessage"], None],
        body: Union[EventMessageBody, Exception],
        # 是否实时处理
        is_real_time: bool = False,
        # 是否异步处理
        is_async: bool = True,
        is_wait_async: bool = False,
        wait_async_timeout: float = 0.0,
    ):
        self.request_id = request_id
        self.result_consumer = result_consumer
        self.body = body
        # TODO: 用来判断是否是实时处理，如果是实时处理，需要立即返回结果
        self.is_real_time = is_real_time
        self.is_async = is_async
        self.is_wait_async = is_wait_async
        self.wait_async_timeout = wait_async_timeout
        # 时间戳，用于判断处理是否超时
        self.timestamp = int(time.time() * 1000)

    def copy(self):
        return EventMessage(
            self.request_id,
            self.result_consumer,
            self.body,
            self.is_real_time,
            self.is_async,
            self.is_wait_async,
            self.wait_async_timeout,
        )
