#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 18:09:13
"""

import multiprocessing as mp
import os
import time
from multiprocessing.pool import Pool
from typing import Optional, Union

import numpy as np

import config
from face_recognition import ArcFaceRecognizer
from store.local_store import LocalStore
from utils.hash_util import calculate_md5
from utils.logger_util import infer_logger, ms_logger
from utils.uuid_util import get_uuid

from .face_detection_service import (
    destroy_detector,
    detector_detect_faces,
    load_detector,
)
from .store.face_recognition_store import FaceRecognitionStore

# 主进程中的全局变量
recognizer_pool: Optional[Pool] = None
face_recognition_store: FaceRecognitionStore

# 进程池每个进程中的全局变量
recognizer: Optional[ArcFaceRecognizer] = None

def load_recognizer():
    global recognizer_pool
    if recognizer_pool is None:
        mp.set_start_method("spawn", True)
        recognizer_pool = mp.Pool(
            config.model_service_server_face_recognize_max_workers,
            initializer=init_recognizer_pool_process,
        )
    ms_logger.info("face recognizer pool loaded")
    return recognizer_pool


def load_face_recognition_store():
    global face_recognition_store
    # TODO: 需要使用 RedisStore，保证多个实例的数据一致性
    face_recognition_store = FaceRecognitionStore(
        LocalStore.create, max_face_count=1000
    )
    register_faces()
    return face_recognition_store


# 初始化进程池的进程
def init_recognizer_pool_process():
    global recognizer
    if recognizer is not None:
        return recognizer
    recognizer = ArcFaceRecognizer(
        trained_model=config.face_recognize_model,
        network=config.face_recognize_network,
        cpu=config.face_recognize_cpu,
    )
    ms_logger.info("face recognizer worker loaded")
    return recognizer


# 以下是进程池中的函数
def recognizer_gen_feat(
    img: Union[str, np.ndarray], face_lmks: Optional[np.ndarray] = None
) -> np.ndarray:
    if recognizer is None:
        raise ValueError("Recognizer is not loaded")
    return recognizer.gen_feat(img, face_lmks)


def recognizer_calc_similarity_batch(
    feat1: np.ndarray, feat2: np.ndarray
) -> np.ndarray:
    if recognizer is None:
        raise ValueError("Recognizer is not loaded")
    return recognizer.calc_similarity_batch(feat1, feat2)


# 以下是主进程中的函数

def register_faces():
    if not os.path.exists(config.face_recognize_register_path):
        return
    load_detector()
    for root, dirs, files in os.walk(config.face_recognize_register_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                label = ".".join(os.path.basename(file).split(".")[:-1])

                img, face_bbox, face_lmks = detect_faces(img_path)
                if img is None:
                    continue

                face = img[face_bbox[1] : face_bbox[3], face_bbox[0] : face_bbox[2]]
                face_lmks -= np.array([face_bbox[0], face_bbox[1]])

                register_face(face, face_lmks, label)
                ms_logger.info(f"register face {label} successfully")

    # 服务没有开启人脸检测服务的话，使用完就关闭
    if not config.face_detection_enabled:
        destroy_detector()


def register_face(
    face: np.ndarray,
    face_lmks: np.ndarray,
    label: str,
):
    global recognizer_pool
    if recognizer_pool is None:
        recognizer_pool = load_recognizer()
    feat = recognizer_pool.apply(
        recognizer_gen_feat,
        (
            face,
            face_lmks,
        ),
    )
    face_recognition_store.save_feat(label, feat)


def recognize_faces(
    face: np.ndarray,
    face_lmks: np.ndarray,
) -> str:
    if recognizer_pool is None:
        raise ValueError("Recognizer pool is not loaded")
    curr_time = time.time()
    feat = recognizer_pool.apply(
        recognizer_gen_feat,
        (
            face,
            face_lmks,
        ),
    )
    infer_logger.debug(
        f"Face recognize gen_feature cost: {time.time() - curr_time:.4f}s"
    )
    lib_feat, lib_labels = get_lib_feat_and_labels()
    if lib_feat.shape[0] == 0:
        return create_new_label(feat)
    curr_time = time.time()
    sim = recognizer_pool.apply(
        recognizer_calc_similarity_batch, (np.expand_dims(feat, axis=0), lib_feat)
    )
    infer_logger.debug(
        f"Face recognize calc_similarity_batch cost: {time.time() - curr_time:.4f}s"
    )
    sim = sim[0]
    max_idx = np.argmax(sim)
    max_sim = sim[max_idx]
    if max_sim < config.face_recognize_sim_threshold:
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


def detect_faces(
    image_path: str,
) -> tuple[Optional[np.ndarray], tuple[int, int, int, int], np.ndarray]:
    face_detector_pool = load_detector()
    face_dets, img, *_ = face_detector_pool.apply(detector_detect_faces, (image_path,))
    face_dets = list(
        filter(
            lambda x: x[4] > config.face_detection_confidence_threshold,
            face_dets,
        )
    )
    if len(face_dets) == 0:
        return None, (0, 0, 0, 0), np.array([])
    face_det = face_dets[0]

    x1 = face_det[0]
    y1 = face_det[1]
    x2 = face_det[2]
    y2 = face_det[3]
    face_bbox: tuple[int, int, int, int] = (x1, y1, x2, y2)

    left_eye_x = face_det[7]
    left_eye_y = face_det[8]
    right_eye_x = face_det[9]
    right_eye_y = face_det[10]
    nose_x = face_det[11]
    nose_y = face_det[12]
    left_mouth_x = face_det[13]
    left_mouth_y = face_det[14]
    right_mouth_x = face_det[15]
    right_mouth_y = face_det[16]
    face_lmks = np.array(
        [
            [left_eye_x, left_eye_y],
            [right_eye_x, right_eye_y],
            [nose_x, nose_y],
            [left_mouth_x, left_mouth_y],
            [right_mouth_x, right_mouth_y],
        ]
    ).astype(np.float32)

    return img, face_bbox, face_lmks
