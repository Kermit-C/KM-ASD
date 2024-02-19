#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 模型初始化
@Author: Kermit
@Date: 2024-02-19 12:50:53
"""

import config
from manager.grpc_server_manager import start_server
from service.face_detection_service import load_detector
from service.face_recognition_service import load_recognizer
from service.speaker_verification_service import load_verificator


def initialize_models(model_type: str):
    if model_type == "face_detection":
        return load_detector()
    elif model_type == "face_recognition":
        return load_recognizer()
    elif model_type == "speaker_verification":
        return load_verificator()
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def initialize_model_service(wait_for_termination=True):
    start_server(
        config.model_service_server_max_workers,
        config.model_service_server_grpc_port,
        wait_for_termination,
    )