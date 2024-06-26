#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-16 21:47:05
"""


from threading import RLock
from typing import Any


class LocalCache:
    def __init__(self) -> None:
        self.cache = {}

    def has(self, key):
        return key in self.cache

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        self.cache[key] = value

    def get_all_entries(self) -> list[tuple[Any, Any]]:
        return list(self.cache.items())


class LocalLruCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        self.lock = RLock()

    def has(self, key):
        return key in self.cache

    def get(self, key):
        with self.lock:
            if key in self.cache:
                # 将访问的键移到最前面
                self.order.remove(key)
                self.order.insert(0, key)
                return self.cache[key]
            else:
                return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                # 如果键已存在，将其移到最前面
                self.order.remove(key)
            elif len(self.cache) >= self.capacity:
                # 删除最近最少使用的键
                old_key = self.order.pop()
                del self.cache[old_key]

            self.cache[key] = value
            self.order.insert(0, key)

    def get_all_entries(self) -> list[tuple[Any, Any]]:
        return list(self.cache.items())
