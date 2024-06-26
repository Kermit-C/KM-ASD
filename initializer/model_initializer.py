#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 模型初始化
@Author: Kermit
@Date: 2024-02-19 12:50:53
"""

import config
from manager.grpc_server_manager import start_server
from manager.metric_manager import init as init_metric_manager
from manager.system_metric_manager import init as init_system_metric_manager
from manager.system_metric_manager import (
    load_metric_collector as load_system_metric_collector,
)
from service.asd_service import load_asd_metric, load_asd_store
from service.asd_service import load_detector as load_asd_detector
from service.face_detection_service import load_detector as load_face_detector
from service.face_detection_service import load_face_detection_metric
from service.face_recognition_service import (
    load_face_recognition_metric,
    load_face_recognition_store,
    load_recognizer,
)
from service.speaker_verification_service import (
    load_speaker_verification_metric,
    load_speaker_verification_store,
    load_verificator,
)


def initialize_models(model_type: str):
    init_metric_manager()
    load_system_metric_collector()
    init_system_metric_manager()
    if model_type == "asd":
        load_asd_metric()
        load_asd_store()
        load_asd_detector()
    elif model_type == "face_detection":
        load_face_detection_metric()
        load_face_detector()
    elif model_type == "face_recognition":
        load_face_recognition_metric()
        load_face_recognition_store()
        load_recognizer()
    elif model_type == "speaker_verification":
        load_speaker_verification_metric()
        load_speaker_verification_store()
        load_verificator()
    else:
        raise ValueError(f"Invalid model type: {model_type}")


def initialize_model_service(wait_for_termination=True):
    start_server(
        config.model_service_server_max_workers,
        config.model_service_server_grpc_port,
        wait_for_termination,
    )
