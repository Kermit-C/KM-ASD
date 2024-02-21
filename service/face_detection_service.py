#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 17:15:19
"""


from typing import Any, Optional

import numpy as np

import config
from face_detection.retinaface_torch import RetinaFaceDetector

detector: Optional[RetinaFaceDetector] = None

def load_detector():
    global detector
    if detector is not None:
        return detector
    detector = RetinaFaceDetector(
        trained_model=config.face_detection_model,
        network=config.face_detection_network,
        cpu=config.face_detection_cpu,
    )
    return detector


def destroy_detector():
    # TODO
    pass


def detect_faces(image: np.ndarray) -> list[dict[str, Any]]:
    if detector is None:
        raise ValueError("Detector is not loaded")
    face_dets, _ = detector.detect_faces(
        image_or_image_path=image,
    )
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
