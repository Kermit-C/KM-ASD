#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线执行器
@Author: chenkeming
@Date: 2024-02-20 22:59:07
"""

import asyncio
import threading
from concurrent import futures
from typing import Callable

import config

excecuter: futures.ThreadPoolExecutor
event_loop: asyncio.AbstractEventLoop
event_loop_thread: threading.Thread

def init():
    global excecuter
    global event_loop
    global event_loop_thread
    excecuter = futures.ThreadPoolExecutor(
        max_workers=config.event_bus_executor_max_workers,
        thread_name_prefix=config.event_bus_executor_thread_name_prefix,
    )
    event_loop = asyncio.new_event_loop()
    event_loop_thread = threading.Thread(target=event_loop.run_forever, daemon=True)
    event_loop_thread.start()


def submit(task: Callable, *args, **kwargs) -> futures.Future:
    return excecuter.submit(task, *args, **kwargs)


def get_event_loop():
    return event_loop


def check_running_loop() -> bool:
    """检查当前线程是否是在事件循环中"""
    return event_loop_thread.ident == threading.current_thread().ident
