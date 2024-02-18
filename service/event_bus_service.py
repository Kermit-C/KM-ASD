#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线服务
@Author: Kermit
@Date: 2024-02-18 17:16:07
"""

import os
from concurrent import futures
from typing import Any, Optional

import cv2
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
            result_video_path = _render_video(
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


def _render_video(
    request_id: str,
    render_video_path: str,
    video_frames: list[np.ndarray],
    audio_path: str,
) -> str:
    output_path = os.path.join(render_video_path, f"{request_id}.mp4")
    frame_height, frame_width, _ = video_frames[0].shape
    # mp4v: MPEG-4 Part 2 video codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    for frame in video_frames:
        video_writer.write(frame)

    video_writer.release()
    return output_path


def call_face_detection(frame: np.ndarray) -> list[dict[str, Any]]:
    # TODO: 调用人脸检测服务
    return []


def call_face_recognition(face: np.ndarray, face_lmks: np.ndarray) -> str:
    # TODO: 调用人脸识别服务
    return "label"


def call_speaker_verification(audio: torch.Tensor) -> str:
    # TODO: 调用说话人验证服务
    return "label"
