#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 模型服务入口
@Author: chenkeming
@Date: 2024-02-10 15:51:57
"""

import config
import initializer

if config.asd_enabled:
    initializer.initialize_models("asd")
if config.face_detection_enabled:
    initializer.initialize_models("face_detection")
if config.face_recognize_enabled:
    initializer.initialize_models("face_recognition")
if config.speaker_verificate_enabled:
    initializer.initialize_models("speaker_verification")

initializer.initialize_model_service(wait_for_termination=True)
