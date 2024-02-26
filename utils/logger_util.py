#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-26 16:14:06
"""

import time


class Logger:
    def __init__(self, scope: str, is_show_time: bool = True) -> None:
        self.scope = scope
        self.is_show_time = is_show_time

    def info(self, *msg: object) -> None:
        print(
            *(
                [f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]']
                if self.is_show_time
                else []
            ),
            "[INFO]",
            f"[{self.scope}]",
            *msg,
        )

    def error(self, *msg: object) -> None:
        print(
            *(
                [f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]']
                if self.is_show_time
                else []
            ),
            "[ERROR]",
            f"[{self.scope}]",
            *msg,
        )

    def debug(self, *msg: object) -> None:
        print(
            *(
                [f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]']
                if self.is_show_time
                else []
            ),
            "[DEBUG]",
            f"[{self.scope}]",
            *msg,
        )

    def warning(self, *msg: object) -> None:
        print(
            *(
                [f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]']
                if self.is_show_time
                else []
            ),
            "[WARNING]",
            f"[{self.scope}]",
            *msg,
        )


eb_logger = Logger("EventBus", is_show_time=True)
ms_logger = Logger("ModelService", is_show_time=True)
