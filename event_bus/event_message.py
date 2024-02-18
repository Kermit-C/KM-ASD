#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线消息
@Author: chenkeming
@Date: 2024-02-09 17:16:48
"""

import time
from typing import Callable


class EventMessageBody:
    pass


class EventMessage:

    def __init__(
        self,
        request_id: str,
        result_consumer: Callable[["EventMessage"], None],
        body: EventMessageBody,
    ):
        self.request_id = request_id
        self.result_consumer = result_consumer
        self.body = body
        # 时间戳，用于判断处理是否超时
        self.timestamp = int(time.time() * 1000)

    def copy(self):
        return EventMessage(self.request_id, self.result_consumer, self.body)
