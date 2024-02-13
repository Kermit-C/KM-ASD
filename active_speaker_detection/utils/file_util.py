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


def postprocess_speech_label(speech_label):
    speech_label = int(speech_label)
    if speech_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
        speech_label = 0
    return speech_label


def postprocess_entity_label(entity_label):
    entity_label = int(entity_label)
    if entity_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
        entity_label = 0
    return entity_label


def csv_to_list(csv_path: str) -> List[List[str]]:
    """读取 csv 文件，返回 list 格式的数据"""
    as_list = None
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list
