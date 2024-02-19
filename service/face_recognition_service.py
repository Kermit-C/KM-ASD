#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 18:09:13
"""

from typing import Optional

import numpy as np

import config
from face_recognition import ArcFaceRecognizer

recognizer: ArcFaceRecognizer

def load_recognizer():
    global recognizer
    recognizer = ArcFaceRecognizer(
        trained_model=config.face_recognize_model,
        network=config.face_recognize_network,
        cpu=config.face_recognize_cpu,
    )
    return recognizer


def recognize_faces(
    face: np.ndarray,
    face_lmks: np.ndarray,
) -> str:
    feat = recognizer.gen_feat(
        img=face,
        face_lmks=face_lmks,
    )
    lib_feat, lib_labels = get_lib_feat_and_labels()
    sim = recognizer.calc_similarity_batch(np.expand_dims(feat, axis=0), lib_feat)
    sim = sim[0]
    max_idx = np.argmax(sim)
    max_sim = sim[max_idx]
    if max_sim < config.face_recognize_sim_threshold:
        return create_new_label(feat)
    label = lib_labels[max_idx]
    return label


def get_lib_feat_and_labels() -> tuple[np.ndarray, list[str]]:
    # TODO
    return np.array([]), []


def create_new_label(feature: np.ndarray) -> str:
    # TODO
    return "new_label"
