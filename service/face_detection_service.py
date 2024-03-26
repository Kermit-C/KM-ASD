#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 17:15:19
"""


import multiprocessing as mp
from multiprocessing.pool import Pool
from typing import Any, Optional, Union

import cv2
import numpy as np

import config
from face_detection.retinaface_torch import RetinaFaceDetector
from manager.metric_manager import MetricCollector, create_collector
from utils.logger_util import infer_logger, ms_logger

# 主进程中的全局变量
detector_pool: Optional[Pool] = None
metric_collector_of_forward_duration: MetricCollector
metric_collector_of_misc_duration: MetricCollector

# 进程池每个进程中的全局变量
detector: Optional[RetinaFaceDetector] = None

def load_detector():
    global detector_pool
    if detector_pool is None:
        mp.set_start_method("spawn", True)
        detector_pool = mp.Pool(
            config.model_service_server_face_detection_max_workers,
            initializer=init_detector_pool_process,
        )
    ms_logger.info("face detector pool loaded")
    return detector_pool


def load_face_detection_metric():
    global metric_collector_of_forward_duration
    global metric_collector_of_misc_duration
    metric_collector_of_forward_duration = create_collector(
        f"model_service_face_detection_forward_duration"
    )
    metric_collector_of_misc_duration = create_collector(
        f"model_service_face_detection_misc_duration"
    )


# 初始化进程池的进程
def init_detector_pool_process():
    global detector
    if detector is not None:
        return detector
    detector = RetinaFaceDetector(
        trained_model=config.face_detection_model,
        network=config.face_detection_network,
        cpu=config.face_detection_cpu,
    )
    ms_logger.info("face detector worker loaded")
    return detector


def destroy_detector():
    # TODO
    pass


# 以下是进程池中的函数
def detector_detect_faces(
    image_or_image_path: Union[cv2.typing.MatLike, np.ndarray, str],
    save_path: Optional[str] = None,
):
    if detector is None:
        raise ValueError("Detector is not loaded")
    return detector.detect_faces(image_or_image_path, save_path)


# 以下是主进程中的函数


def detect_faces(image: np.ndarray) -> list[dict[str, Any]]:
    if detector_pool is None:
        raise ValueError("Detector pool is not loaded")
    face_dets, _, forward_time, misc_time = detector_pool.apply(
        detector_detect_faces, (image,)
    )
    infer_logger.debug(
        "Face detection forward time: {:.4f}s misc: {:.4f}s".format(
            forward_time, misc_time
        )
    )
    metric_collector_of_forward_duration.collect(forward_time)
    metric_collector_of_misc_duration.collect(misc_time)

    face_dets = filter(
        lambda x: x[4] > config.face_detection_confidence_threshold, face_dets
    )
    return [
        {
            "bbox": {
                "x1": face_det[0],
                "y1": face_det[1],
                "x2": face_det[2],
                "y2": face_det[3],
                "width": face_det[5],
                "height": face_det[6],
            },
            "confidence": face_det[4],
            "landmarks": {
                "left_eye": {
                    "x": face_det[7],
                    "y": face_det[8],
                },
                "right_eye": {
                    "x": face_det[9],
                    "y": face_det[10],
                },
                "nose": {
                    "x": face_det[11],
                    "y": face_det[12],
                },
                "left_mouth": {
                    "x": face_det[13],
                    "y": face_det[14],
                },
                "right_mouth": {
                    "x": face_det[15],
                    "y": face_det[16],
                },
            },
        }
        for face_det in face_dets
    ]
