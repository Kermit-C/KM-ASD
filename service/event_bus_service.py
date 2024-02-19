#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线服务
@Author: Kermit
@Date: 2024-02-18 17:16:07
"""

import json
import pickle
from concurrent import futures
from typing import Any, Optional

import grpc
import numpy as np
import torch

import config
from event_bus import get_publisher
from event_bus.event_message import EventMessage, EventMessageBody
from event_bus.message_body.audio_to_pcm_message_body import AudioToPcmMessageBody
from event_bus.message_body.result_message_body import ResultMessageBody
from event_bus.message_body.video_to_frame_message_body import VideoToFrameMessageBody
from event_bus.processor.audio_to_pcm_processor import AudioToPcmProcessor
from event_bus.processor.video_to_frame_processor import VideoToFrameProcessor
from grpc_service import model_service_pb2, model_service_pb2_grpc
from utils.io_util import render_video
from utils.uuid_util import get_uuid

_consume_cache = {}


def process(
    request_id: str,
    video_path: str,
    audio_path: str,
    render_video_path: str,
    timeout_second: Optional[int] = None,
) -> str:
    """处理视频和音频，返回处理后的视频路径"""
    v_body = VideoToFrameMessageBody(video_path=video_path)
    a_body = AudioToPcmMessageBody(audio_path=audio_path)

    result_future: futures.Future[str] = futures.Future()

    def _consume_result(message: EventMessage):
        request_id = message.request_id
        message_body = message.body
        assert isinstance(message_body, ResultMessageBody)
        if request_id not in _consume_cache.keys():
            _consume_cache[request_id] = {
                "video_frames": [],
                "video_fps": message_body.video_fps,
                "video_frame_count": message_body.video_frame_count,
            }

        _consume_cache[request_id]["video_frames"].append(message_body.frame)
        if (
            len(_consume_cache[request_id]["video_frames"])
            == _consume_cache[request_id]["video_frame_count"]
        ):
            # 完成了视频帧的处理就渲染视频
            result_video_path = render_video(
                request_id,
                render_video_path,
                _consume_cache[request_id]["video_frames"],
                audio_path,
            )
            result_future.set_result(result_video_path)

    result_consumer = _consume_result
    v_message = EventMessage(request_id, result_consumer, v_body)
    a_message = EventMessage(request_id, result_consumer, a_body)

    publisher = get_publisher(config.event_bus["publisher"]["name"])
    publisher.publish(
        config.event_bus["processors"][VideoToFrameProcessor]["topic"], v_message
    )
    publisher.publish(
        config.event_bus["processors"][AudioToPcmProcessor]["topic"], a_message
    )

    return result_future.result(timeout=timeout_second)


def call_face_detection(frame: np.ndarray) -> list[dict[str, Any]]:
    # 调用人脸检测服务
    # TODO: 实现负载均衡
    server_host = "localhost:50051"

    with grpc.insecure_channel(server_host) as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)

        frame_bytes = pickle.dumps(frame)
        response: model_service_pb2.FaceDetectionResponse = stub.call_face_detection(
            model_service_pb2.FaceDetectionRequest(
                meta=model_service_pb2.RequestMetaData(request_id=get_uuid()),  # type: ignore
                face_image=frame_bytes,
            )
        )
        face_dets_json: str = response.face_dets_json  # type: ignore
        face_dets: list[dict[str, Any]] = json.loads(face_dets_json)

        return face_dets


def call_face_recognition(face: np.ndarray, face_lmks: np.ndarray) -> str:
    # 调用人脸识别服务
    # TODO: 实现负载均衡
    server_host = "localhost:50051"

    with grpc.insecure_channel(server_host) as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)

        face_bytes = pickle.dumps(face)
        face_lmks_bytes = pickle.dumps(face_lmks)
        response: model_service_pb2.FaceRecognitionResponse = (
            stub.call_face_recognition(
                model_service_pb2.FaceRecognitionRequest(
                    meta=model_service_pb2.RequestMetaData(request_id=get_uuid()),  # type: ignore
                    face_image=face_bytes,
                    face_lmks=face_lmks_bytes,
                )
            )
        )
        label: str = response.label  # type: ignore

        return label


def call_speaker_verification(audio: torch.Tensor) -> str:
    # 调用说话人验证服务
    # TODO: 实现负载均衡
    server_host = "localhost:50051"

    with grpc.insecure_channel(server_host) as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)

        audio_bytes = pickle.dumps(audio.numpy())
        response: model_service_pb2.SpeakerVerificationResponse = (
            stub.call_speaker_verification(
                model_service_pb2.SpeakerVerificationRequest(
                    meta=model_service_pb2.RequestMetaData(request_id=get_uuid()),  # type: ignore
                    voice_data=audio_bytes,
                )
            )
        )
        label: str = response.label  # type: ignore

        return label
