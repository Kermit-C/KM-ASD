#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-16 18:34:00
"""

import hashlib


def calculate_md5(text: str) -> int:
    """计算字符串的 MD5 值，返回十进制整数"""
    md5_hash = hashlib.md5(text.encode())
    md5_hex = md5_hash.hexdigest()
    md5_decimal = int(md5_hex, 16)
    return md5_decimal
