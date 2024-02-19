#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 主入口
@Author: chenkeming
@Date: 2024-02-10 15:51:57
"""

import config
import initializer
from service.event_bus_service import process
from utils.io_util import extract_audio_track
from utils.uuid_util import get_uuid

initializer.init_event_bus()
initializer.initialize_model_service(wait_for_termination=False)
initializer.initialize_models("face_detection")
initializer.initialize_models("face_recognition")
initializer.initialize_models("speaker_verification")


def process_fn(
    video_path: str,
) -> dict:
    request_id = get_uuid()
    audio_path = extract_audio_track(video_path)
    render_video_path = config.render_video_path
    result_path = process(request_id, video_path, audio_path, render_video_path)
    return {"result_path": result_path}
