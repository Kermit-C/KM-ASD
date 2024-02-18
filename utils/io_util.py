#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-18 22:33:44
"""

import os
import subprocess

import cv2
import numpy as np

import config


def render_video(
    request_id: str,
    render_video_path: str,
    video_frames: list[np.ndarray],
    audio_path: str,
) -> str:
    # TODO: 把声音也渲染进去
    output_path = os.path.join(render_video_path, f"{request_id}.mp4")
    frame_height, frame_width, _ = video_frames[0].shape
    # mp4v: MPEG-4 Part 2 video codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    for frame in video_frames:
        video_writer.write(frame)

    video_writer.release()
    return output_path


def extract_audio_track(video_path: str) -> str:
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    command = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar %d %s" % (
        video_path,
        config.extract_audio_track_sample_rate,
        audio_path,
    )
    subprocess.call(command, shell=True, stdout=None)
    return audio_path
