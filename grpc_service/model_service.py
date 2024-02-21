#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 模型 grpc 服务
@Author: Kermit
@Date: 2024-02-18 17:15:19
"""

import json
import pickle
from threading import Semaphore

import grpc
import numpy as np

import config
from service.face_detection_service import detect_faces
from service.face_recognition_service import recognize_faces
from service.speaker_verification_service import verify_speakers
from utils.uuid_util import get_uuid

from . import model_service_pb2, model_service_pb2_grpc


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self):
        self.face_detection_semaphore = Semaphore(
            config.model_service_server_face_detection_max_workers
        )
        self.face_recognition_semaphore = Semaphore(
            config.model_service_server_face_recognize_max_workers
        )
        self.speaker_verification_semaphore = Semaphore(
            config.model_service_server_speaker_verificate_max_workers
        )

    def call_face_detection(
        self,
        request: model_service_pb2.FaceDetectionRequest,
        context: grpc.aio.ServicerContext,
    ) -> model_service_pb2.FaceDetectionResponse:
        request_id = request.meta.request_id  # type: ignore
        face_image = request.face_image  # type: ignore
        face_image_np: np.ndarray = pickle.loads(face_image)

        is_acquired = self.face_detection_semaphore.acquire(
            blocking=True,
            timeout=config.model_service_server_face_detection_worker_wait_timeout,
        )
        if not is_acquired:
            raise Exception("Face detection worker is busy")
        try:
            face_dets = detect_faces(face_image_np)
        finally:
            self.face_detection_semaphore.release()

        return model_service_pb2.FaceDetectionResponse(
            meta=model_service_pb2.ResponseMetaData(
                response_id=get_uuid(), request_id=request_id
            ),  # type: ignore
            face_dets_json=json.dumps(face_dets),
        )

    def call_face_recognition(
        self,
        request: model_service_pb2.FaceRecognitionRequest,
        context: grpc.aio.ServicerContext,
    ) -> model_service_pb2.FaceRecognitionResponse:
        request_id = request.meta.request_id  # type: ignore
        face_image = request.face_image  # type: ignore
        face_lmks = request.face_lmks  # type: ignore
        face_image_np: np.ndarray = pickle.loads(face_image)
        face_lmks_np: np.ndarray = pickle.loads(face_lmks)

        is_acquired = self.face_recognition_semaphore.acquire(
            blocking=True,
            timeout=config.model_service_server_face_recognize_worker_wait_timeout,
        )
        if not is_acquired:
            raise Exception("Face recognition worker is busy")
        try:
            label = recognize_faces(face_image_np, face_lmks_np)
        finally:
            self.face_recognition_semaphore.release()

        return model_service_pb2.FaceRecognitionResponse(
            meta=model_service_pb2.ResponseMetaData(
                response_id=get_uuid(), request_id=request_id
            ),  # type: ignore
            label=label,
        )

    def call_speaker_verification(
        self,
        request: model_service_pb2.SpeakerVerificationRequest,
        context: grpc.aio.ServicerContext,
    ) -> model_service_pb2.SpeakerVerificationResponse:
        request_id = request.meta.request_id  # type: ignore
        voice_data = request.voice_data  # type: ignore
        voice_data_np: np.ndarray = pickle.loads(voice_data)

        is_acquired = self.speaker_verification_semaphore.acquire(
            blocking=True,
            timeout=config.model_service_server_speaker_verificate_worker_wait_timeout,
        )
        if not is_acquired:
            raise Exception("Speaker verification worker is busy")
        try:
            label = verify_speakers(voice_data_np)
        finally:
            self.speaker_verification_semaphore.release()

        return model_service_pb2.SpeakerVerificationResponse(
            meta=model_service_pb2.ResponseMetaData(
                response_id=get_uuid(), request_id=request_id
            ),  # type: ignore
            label=label,
        )
