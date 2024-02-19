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
from store.local_store import LocalStore
from utils.hash_util import calculate_md5
from utils.uuid_util import get_uuid

from .store.face_recognition_store import FaceRecognitionStore

recognizer: ArcFaceRecognizer
face_recognition_store: FaceRecognitionStore

def load_recognizer():
    global recognizer
    recognizer = ArcFaceRecognizer(
        trained_model=config.face_recognize_model,
        network=config.face_recognize_network,
        cpu=config.face_recognize_cpu,
    )
    return recognizer


def load_face_recognition_store():
    global face_recognition_store
    face_recognition_store = FaceRecognitionStore(
        LocalStore.create, max_face_count=1000
    )
    # TODO: 实现注册人脸
    return face_recognition_store


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
        # TODO: 存在上一次的新 label 未保存完，这里又存了新的 label 的问题，考虑用 label 关联解决
        return create_new_label(feat)
    label = lib_labels[max_idx]
    return label


def get_lib_feat_and_labels() -> tuple[np.ndarray, list[str]]:
    labels, feat_list = face_recognition_store.get_all_feats()
    feat = np.array(feat_list)
    return feat, labels


def create_new_label(feature: np.ndarray) -> str:
    label = str(calculate_md5(get_uuid()) % 1000000000)
    face_recognition_store.save_feat(label, feature)
    return label
