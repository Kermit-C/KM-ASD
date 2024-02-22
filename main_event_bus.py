#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 接入服务入口（事件总线）
@Author: chenkeming
@Date: 2024-02-10 15:51:57
"""

import config
import initializer
from service.event_bus_service import process
from utils.io_util import extract_audio_track
from utils.uuid_util import get_uuid

if config.asd_enabled:
    initializer.initialize_models("asd")
if config.face_detection_enabled:
    initializer.initialize_models("face_detection")
if config.face_recognize_enabled:
    initializer.initialize_models("face_recognition")
if config.speaker_verificate_enabled:
    initializer.initialize_models("speaker_verification")
if (
    config.face_detection_enabled
    or config.face_recognize_enabled
    or config.speaker_verificate_enabled
):
    initializer.initialize_model_service(wait_for_termination=False)

initializer.init_event_bus()


def process_fn(
    video_path: str,
) -> dict:
    request_id = get_uuid()
    audio_path = extract_audio_track(video_path)
    render_video_path = config.render_video_path
    result_path = process(request_id, video_path, audio_path, render_video_path)
    return {"result_path": result_path}
