#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-13 16:00:01
"""

import csv
import os
from typing import List

"""
标签：
0: NOT_SPEAKING
1: SPEAKING_AUDIBLE
2: SPEAKING_NOT_AUDIBLE
"""

def postprocess_speech_label(speech_label):
    speech_label = int(speech_label)
    if speech_label == 2:
        # 把说话但没有声音也标记为未说话
        speech_label = 0
    return speech_label


def postprocess_entity_label(entity_label):
    entity_label = int(entity_label)
    if entity_label == 2:
        # 把说话但没有声音也标记为未说话
        entity_label = 0
    return entity_label


def csv_to_list(csv_path: str) -> List[List[str]]:
    """读取 csv 文件，返回 list 格式的数据"""
    as_list = None
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list
