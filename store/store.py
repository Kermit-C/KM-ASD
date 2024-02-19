#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-16 21:52:45
"""


from typing import Any


class Store:

    def has(self, key: str) -> bool:
        raise NotImplementedError

    def get(self, key: str):
        raise NotImplementedError

    def put(self, key: str, value):
        raise NotImplementedError

    def get_all_entries(self) -> list[tuple[Any, Any]]:
        raise NotImplementedError
