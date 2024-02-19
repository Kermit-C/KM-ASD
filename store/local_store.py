#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 本地存储
@Author: Kermit
@Date: 2024-02-16 21:33:29
"""

from .cache import LocalCache, LocalLruCache
from .store import Store


class LocalStore(Store):
    def __init__(self, lru: bool = True, lru_capacity: int = 1000):
        self.lru = lru
        self.lru_capacity = lru_capacity
        if self.lru:
            self.store = LocalLruCache(lru_capacity)
        else:
            self.store = LocalCache()

    def has(self, key):
        return self.store.has(key)

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        self.store.put(key, value)

    def get_all_entries(self):
        return self.store.get_all_entries()

    @staticmethod
    def create(lru: bool = True, lru_capacity: int = 1000):
        return LocalStore(lru, lru_capacity)
