#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-09 17:16:48
"""

import numpy as np

from face_detection import RetinaFaceDetector
from face_recognition import ArcFaceRecognizer


def test_face_detector():
    # detector = RetinaFaceDetector(
    #     trained_model="face_detection/retinaface_weights/Resnet50_Final.pth",
    #     network="resnet50",
    #     cpu=True,
    # )
    detector = RetinaFaceDetector(
        trained_model="face_detection/retinaface_weights/mobilenet0.25_Final.pth",
        network="mobile0.25",
        cpu=False,
    )
    dets = detector.detect_faces(
        image_or_image_path="/hdd1/ckm/asd/face_detection/retinaface/curve/test.jpg",
        save_path="./tmp/test-face-detector.jpg",
    )
    print(dets)


def test_face_recognition():
    recognizer = ArcFaceRecognizer(
        trained_model="face_recognition/arcface_weights/ms1mv3_r18_backbone.pth",
        network="r18",
        cpu=False,
    )
    detector = RetinaFaceDetector(
        trained_model="face_detection/retinaface_weights/mobilenet0.25_Final.pth",
        network="mobile0.25",
        cpu=False,
    )

    face1 = "/hdd1/ckm/asd/tmp/tmp_faces/900.0.jpg"
    face2 = "/hdd1/ckm/asd/tmp/tmp_faces/901.91.jpg"

    [dets1, *_] = detector.detect_faces(
        image_or_image_path=face1,
    )
    [dets2, *_] = detector.detect_faces(
        image_or_image_path=face2,
    )
    if dets1 is None or dets2 is None:
        print("No face detected!")
        return
    feat1 = recognizer.gen_feat(
        img=face1,
        face_lmks=np.array(dets1[5:15]).reshape((5, 2)).astype(np.float32),
    )
    feat2 = recognizer.gen_feat(
        img=face2,
        face_lmks=np.array(dets2[5:15]).reshape((5, 2)).astype(np.float32),
    )
    print(recognizer.calc_similarity(feat1, feat2))


if __name__ == "__main__":
    # test_face_detector()
    test_face_recognition()
