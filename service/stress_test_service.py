#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-03-28 17:56:51
"""

import os
import random
from concurrent import futures

import config
from utils.io_util import extract_audio_track
from utils.logger_util import st_logger

from .event_bus_service import process

excecuter: futures.ThreadPoolExecutor

video_path_set = set()
video_list: list[dict] = []


def _init_executor():
    global excecuter
    excecuter = futures.ThreadPoolExecutor(
        max_workers=config.stress_test_executor_workers,
        thread_name_prefix=config.stress_test_executor_thread_name_prefix,
    )


def _init_video_path_set():
    video_path_set.clear()
    if not os.path.exists(config.stress_test_video_dir):
        raise ValueError(f"Video dir {config.stress_test_video_dir} not exists")

    for root, dirs, files in os.walk(config.stress_test_video_dir):
        for file in files:
            if (
                file.endswith(".mp4")
                or file.endswith(".avi")
                or file.endswith(".flv")
                or file.endswith(".mkv")
                or file.endswith(".mov")
            ):
                video_path_set.add(os.path.join(root, file))

    if len(video_path_set) == 0:
        raise ValueError(f"No video files in {config.stress_test_video_dir}")


def _init_video_list():
    global video_list
    video_path_list = list(video_path_set)
    video_list = [
        {"video_path": video_path, "audio_path": extract_audio_track(video_path)}
        for video_path in video_path_list
    ]


def _excute(request_id: str):
    while True:
        video = random.choice(video_list)
        process(
            request_id,
            video["video_path"],
            video["audio_path"],
            config.render_video_path,
        )
        st_logger.info(f"Request {request_id} process video {video['video_path']}")


def start(wait_for_termination: bool = True):
    _init_executor()
    _init_video_path_set()
    _init_video_list()
    for i in range(config.stress_test_executor_workers):
        request_id = f"stress_test_{i}"
        excecuter.submit(_excute, request_id)

    if wait_for_termination:
        excecuter.shutdown(wait=True)
