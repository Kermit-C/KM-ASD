#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import math
from typing import List, Tuple

# def get_spatial_connection_pattern(ctx_size: int, num_graphs: int):
#     """
#     Deprecated
#     根据上下文大小和图的数量，生成空间连接模式
#     同一个上下文图（同一时刻）中，音频节点连接其他的所有节点，视频节点连接其他的所有节点
#     """
#     cp = {}
#     cp["src"] = []
#     cp["dst"] = []

#     # 先自己连接自己
#     for g in range(num_graphs):
#         graph_offset = g * (ctx_size + 1)
#         for s in range(ctx_size + 1):
#             cp["src"].append(graph_offset + s)
#             cp["dst"].append(graph_offset + s)

#     # 然后音频节点连接其他的所有
#     for g in range(num_graphs):
#         graph_offset = g * (ctx_size + 1)
#         for s in range(1, ctx_size + 1):
#             cp["src"].append(graph_offset + 0)
#             cp["dst"].append(graph_offset + s)

#             cp["src"].append(graph_offset + s)
#             cp["dst"].append(graph_offset + 0)

#     # 视频节点连接其他的所有
#     for g in range(num_graphs):
#         graph_offset = g * (ctx_size + 1)
#         for s in range(1, ctx_size + 1):
#             for d in range(1, ctx_size + 1):
#                 if d != s:
#                     cp["src"].append(graph_offset + s)
#                     cp["dst"].append(graph_offset + d)

#     return cp


# def get_temporal_connection_pattern(ctx_size: int, num_graphs: int):
#     """
#     Deprecated
#     根据上下文大小和图的数量，生成时间连接模式
#     """
#     cp = {}
#     cp["src"] = []
#     cp["dst"] = []

#     # 先自己连接自己
#     for g in range(num_graphs):
#         graph_offset = g * (ctx_size + 1)
#         for s in range(ctx_size + 1):
#             cp["src"].append(graph_offset + s)
#             cp["dst"].append(graph_offset + s)

#     # 视频节点前一个图连接后一个图相同实体的节点
#     for g in range(num_graphs):
#         graph_offset = g * (ctx_size + 1)
#         for s in range(0, ctx_size + 1):

#             if g > 0:
#                 # 连接前一个图
#                 left_graph_offset = (g - 1) * (ctx_size + 1)
#                 cp["src"].append(graph_offset + s)
#                 cp["dst"].append(left_graph_offset + s)

#             if g < num_graphs - 1:
#                 # 连接后一个图
#                 right_graph_offset = (g + 1) * (ctx_size + 1)
#                 cp["src"].append(graph_offset + s)
#                 cp["dst"].append(right_graph_offset + s)

#     return cp


def generate_av_mask(ctx_size: int, total_len: int) -> Tuple[List[int], List[int]]:
    """
    Deprecated
    生成音频和视频的 mask，即音频和视频所在 batch 第零维的索引的列表
    """
    stride = ctx_size + 1
    audio_mask: List[int] = []
    video_mask: List[int] = []
    for i in range(0, total_len):
        if i % stride == 0:
            audio_mask.append(i)
        else:
            video_mask.append(i)
    return audio_mask, video_mask


def generate_temporal_video_mask(ctx_size: int, total_len: int) -> List[int]:
    """
    Deprecated
    生成视频的单个人时序上的 mask，即视频所在 batch 第零维的索引的列表
    """
    stride = ctx_size + 1
    video_mask = [i for i in range(1, total_len, stride)]
    return video_mask


def generate_temporal_video_center_mask(
    ctx_size: int, total_len: int, time_len: int
) -> List[int]:
    """
    Deprecated
    生成视频的单个人时序上的中心帧 mask，即视频所在 batch 第零维的索引的列表
    """
    stride = ctx_size + 1
    video_mask = [
        i + stride * math.floor(time_len / 2)
        for i in range(1, total_len, stride * time_len)
    ]
    return video_mask
