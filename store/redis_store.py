#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-16 21:49:58
"""


from .store import Store


class RedisStore(Store):
    def __init__(self, lru: bool = True, lru_capacity: int = 1000):
        pass

    @staticmethod
    def create(lru: bool = True, lru_capacity: int = 1000):
        return RedisStore(lru, lru_capacity)
