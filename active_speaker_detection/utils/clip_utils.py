#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-05 23:05:15
"""

from typing import List, Tuple


def generate_clip_meta(
    entity_meta_data: List[Tuple[str, str, int]], midone: int, half_clip_size: int
) -> List[Tuple[str, str, int]]:
    """生成一个长度为 half_clip_size*2+1 的单人时间片段
    :param entity_meta_data: 实体元数据
    :param midone: 中心时间戳，entity_meta_data 中的索引
    :param half_clip_size: 时间片段的一半长度
    """
    max_span_left = _get_clip_max_span(entity_meta_data, midone, -1, half_clip_size + 1)
    max_span_right = _get_clip_max_span(entity_meta_data, midone, 1, half_clip_size + 1)

    # 以 midone 为中心，取出时间片段
    clip_data = entity_meta_data[midone - max_span_left : midone + max_span_right + 1]
    clip_data = _extend_clip_data(
        clip_data, max_span_left, max_span_right, half_clip_size
    )
    return clip_data


def _get_clip_max_span(
    entity_meta_data: List[Tuple[str, str, int]], midone: int, direction: int, max: int
):
    idx = 0
    for idx in range(0, max):
        if midone + (idx * direction) < 0:
            return idx - 1
        if midone + (idx * direction) >= len(entity_meta_data):
            return idx - 1

    return idx


def _extend_clip_data(clip_data, max_span_left, max_span_right, half_clip_size):
    """扩展数据，使得数据长度为 half_clip_size*2+1
    如果不够，就复制首尾元素
    """
    if max_span_left < half_clip_size:
        for i in range(half_clip_size - max_span_left):
            clip_data.insert(0, clip_data[0])

    if max_span_right < half_clip_size:
        for i in range(half_clip_size - max_span_right):
            clip_data.insert(-1, clip_data[-1])

    return clip_data
