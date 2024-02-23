#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-05 23:05:15
"""

import random
from typing import List

from PIL import Image
from torch import Tensor
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import hflip


def video_temporal_crop(video_data: List[Image.Image], crop_ratio):
    """对视频进行“随机”裁剪和翻转增强"""
    # random flip
    if bool(random.getrandbits(1)):
        video_data = [s.transpose(Image.FLIP_LEFT_RIGHT) for s in video_data]

    # random crop
    mid = int(len(video_data) / 2)
    width, height = video_data[mid].size
    f = random.uniform(crop_ratio, 1)
    i, j, h, w = RandomCrop.get_params(video_data[mid], output_size=(int(height * f), int(width * f)))  # type: ignore

    video_data = [s.crop(box=(j, i, j+w, i+h)) for s in video_data]
    return video_data
