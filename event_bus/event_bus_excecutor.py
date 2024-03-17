#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线执行器
@Author: chenkeming
@Date: 2024-02-20 22:59:07
"""

from concurrent import futures
from typing import Callable

import config

excecuter: futures.ThreadPoolExecutor


def init():
    global excecuter
    excecuter = futures.ThreadPoolExecutor(
        max_workers=config.event_bus_executor_max_workers,
        thread_name_prefix=config.event_bus_executor_thread_name_prefix,
    )


def submit(task: Callable, *args, **kwargs) -> futures.Future:
    return excecuter.submit(task, *args, **kwargs)


def get_executor():
    return excecuter
