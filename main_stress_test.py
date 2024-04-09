#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 模型服务压力测试入口
@Author: Kermit
@Date: 2024-04-09 18:06:06
"""

import config
import initializer
from service.stress_test_service import start

if __name__ == "__main__":
    if config.asd_enabled:
        initializer.initialize_models("asd")
    if config.face_detection_enabled:
        initializer.initialize_models("face_detection")
    if config.face_recognize_enabled:
        initializer.initialize_models("face_recognition")
    if config.speaker_verificate_enabled:
        initializer.initialize_models("speaker_verification")

    initializer.initialize_model_service(wait_for_termination=False)
    start(wait_for_termination=True)
