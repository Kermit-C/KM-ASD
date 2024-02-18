#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-16 21:52:45
"""


class Store:
    def get(self, key: str):
        raise NotImplementedError

    def put(self, key: str, value):
        raise NotImplementedError
