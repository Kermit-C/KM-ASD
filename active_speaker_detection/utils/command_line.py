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
    parser.add_argument("--name", default="R3D18")
    # 阶段，encoder(训练 encoder)、encoder_gen_emb(用 encoder 生成 emb)、graph(训练图)、end2end(训练 encoder + 图)
    parser.add_argument("--stage", default="end2end")
    # 每刻计算特征的帧数
    parser.add_argument("--frmc", default="13")
    # 上下文大小，每刻的实体数
    parser.add_argument("--ctx", default="3")
    # 图的时间上下文步数，即 clip 数
    parser.add_argument("--nclp", default="7")
    # 图的时间上下文步长，即 clip 之间的间隔，单位为帧
    parser.add_argument("--strd", default="3")
    # 图像大小，将把人脸 crop resize 到这个大小的正方形
    parser.add_argument("--size", default="112")

    return parser


def unpack_command_line_args(args):
    name = args.name
    stage = args.stage
    frames_per_clip = float(args.frmc)
    ctx_size = int(args.ctx)
    n_clips = int(args.nclp)
    strd = int(args.strd)
    img_size = int(args.size)

    return name, stage, frames_per_clip, ctx_size, n_clips, strd, img_size
