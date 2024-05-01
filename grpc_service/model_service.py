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
from service.asd_service import detect_active_speaker
from service.face_detection_service import detect_faces
from service.face_recognition_service import recognize_faces
from service.speaker_verification_service import register_speaker, verify_speakers
from utils.logger_util import ms_logger
from utils.uuid_util import get_uuid

from . import model_service_pb2, model_service_pb2_grpc


class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self):
        self.asd_semaphore = Semaphore(config.model_service_server_asd_max_workers)
        self.face_detection_semaphore = Semaphore(
            config.model_service_server_face_detection_max_workers
        )
        self.face_recognition_semaphore = Semaphore(
            config.model_service_server_face_recognize_max_workers
        )
        self.speaker_verification_semaphore = Semaphore(
            config.model_service_server_speaker_verificate_max_workers
        )

    def call_asd(
        self,
        request: model_service_pb2.AsdRequest,
        context: grpc.ServicerContext,
    ) -> model_service_pb2.AsdResponse:
        request_id = request.meta.request_id  # type: ignore
        video_id = request.request_id  # type: ignore
        frame_count = request.frame_count  # type: ignore
        faces = pickle.loads(request.faces)  # type: ignore
        face_bboxes = pickle.loads(request.face_bboxes)  # type: ignore
        audio = pickle.loads(request.audio)  # type: ignore
        frame_width = request.frame_width  # type: ignore
        frame_height = request.frame_height  # type: ignore
        only_save_frame = request.only_save_frame  # type: ignore

        # ms_logger.debug(f"Received ASD request: {request_id}")

        is_acquired = self.asd_semaphore.acquire(
            blocking=True,
            timeout=config.model_service_server_asd_worker_wait_timeout,
        )
        if not is_acquired:
            # ms_logger.error(f"Active speaker detection worker is busy: {request_id}")
            raise Exception("Active speaker detection worker is busy")

        try:
            if not context.is_active():  # type: ignore
                raise Exception("Face recognition request is cancelled")
            # ms_logger.debug(f"Start processing ASD request: {request_id}")
            is_active_list = detect_active_speaker(
                video_id,
                frame_count,
                faces,
                face_bboxes,
                audio,
                frame_height,
                frame_width,
                only_save_frame,
            )
        finally:
            # ms_logger.debug(f"Finish processing ASD request: {request_id}")
            self.asd_semaphore.release()

        return model_service_pb2.AsdResponse(
            meta=model_service_pb2.ResponseMetaData(
                response_id=get_uuid(), request_id=request_id
            ),  # type: ignore
            is_active=is_active_list,
        )

    def call_face_detection(
        self,
        request: model_service_pb2.FaceDetectionRequest,
        context: grpc.ServicerContext,
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
            if not context.is_active():  # type: ignore
                raise Exception("Face recognition request is cancelled")
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
        context: grpc.ServicerContext,
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
            if not context.is_active():  # type: ignore
                raise Exception("Face recognition request is cancelled")
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
        context: grpc.ServicerContext,
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
            if not context.is_active():  # type: ignore
                raise Exception("Face recognition request is cancelled")
            label = verify_speakers(voice_data_np)
        finally:
            self.speaker_verification_semaphore.release()

        return model_service_pb2.SpeakerVerificationResponse(
            meta=model_service_pb2.ResponseMetaData(
                response_id=get_uuid(), request_id=request_id
            ),  # type: ignore
            label=label,
        )

    def register_speaker(
        self,
        request: model_service_pb2.RegisterSpeakerRequest,
        context: grpc.ServicerContext,
    ) -> model_service_pb2.RegisterSpeakerResponse:
        request_id = request.meta.request_id  # type: ignore
        voice_data = request.voice_data  # type: ignore
        label = request.label  # type: ignore
        voice_data_np: np.ndarray = pickle.loads(voice_data)

        is_acquired = self.speaker_verification_semaphore.acquire(
            blocking=True,
            timeout=config.model_service_server_speaker_verificate_worker_wait_timeout,
        )
        if not is_acquired:
            raise Exception("Speaker verification worker is busy")

        try:
            if not context.is_active():  # type: ignore
                raise Exception("Face recognition request is cancelled")
            label = register_speaker(voice_data_np, label)
        finally:
            self.speaker_verification_semaphore.release()

        return model_service_pb2.RegisterSpeakerResponse(
            meta=model_service_pb2.ResponseMetaData(
                response_id=get_uuid(), request_id=request_id
            ),  # type: ignore
        )
