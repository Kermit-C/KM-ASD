#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import argparse


def get_default_arg_parser():
    parser = argparse.ArgumentParser()
    # 学习率
    parser.add_argument("--lr", default="5e-4")
    # 每刻计算特征的帧数
    parser.add_argument("--frmc", default="13")
    # 上下文大小，每刻的实体数
    parser.add_argument("--ctx", default="2")
    # 图的时间上下文步数，即 clip 数
    parser.add_argument("--nclp", default="7")
    # 图的时间上下文步长，即 clip 之间的间隔，单位为帧
    parser.add_argument("--strd", default="3")
    # 图像大小，将把人脸 crop resize 到这个大小的正方形
    parser.add_argument("--size", default="160")

    return parser


def unpack_command_line_args(args):
    lr_arg = float(args.lr)
    frames_per_clip = float(args.frmc)
    ctx_size = int(args.ctx)
    n_clips = int(args.nclp)
    strd = int(args.strd)
    img_size = int(args.size)

    return lr_arg, frames_per_clip, ctx_size, n_clips, strd, img_size
