#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-09 17:16:48
"""


from face_detection import RetinaFaceDetector


def test_face_detector():
    # detector = RetinaFaceDetector(
    #     trained_model="face_detection/retinaface_weights/Resnet50_Final.pth",
    #     network="resnet50",
    #     cpu=True,
    # )
    detector = RetinaFaceDetector(
        trained_model="face_detection/retinaface_weights/mobilenet0.25_Final.pth",
        network="mobile0.25",
        cpu=True,
    )
    dets = detector.detect_faces(
        image_or_image_path="/hdd1/ckm/asd/face_detection/retinaface/curve/test.jpg",
        save_path="./tmp/test-face-detector.jpg",
    )
    print(dets)


if __name__ == "__main__":
    test_face_detector()
