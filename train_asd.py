#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 训练 ASD 模型
@Author: chenkeming
@Date: 2024-02-16 20:48:53
"""

from active_speaker_detection import asd_config, asd_train

if __name__ == "__main__":
    asd_train.train(asd_config.r3d18_params["param_config"], asd_config.datasets)
