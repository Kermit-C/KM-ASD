#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 训练 ASD 模型
@Author: chenkeming
@Date: 2024-02-16 20:48:53
"""

from active_speaker_detection import asd_r3d18, asd_r3d50

if __name__ == "__main__":
    asd_r3d18.train_asd_r3d18()
    # asd_r3d50.train_asd_r3d50()
