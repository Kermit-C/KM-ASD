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
    video_fps: int,
) -> str:
    # 设置输出视频路径
    output_path = os.path.join(render_video_path, f"{request_id}_no_audio.mp4")
    frame_height, frame_width, _ = video_frames[0].shape
    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    video_writer = cv2.VideoWriter(
        output_path, fourcc, video_fps, (frame_width, frame_height)
    )
    # 写入视频帧
    for frame in video_frames:
        video_writer.write(frame)

    video_writer.release()

    # 使用FFmpeg合并音频和视频
    merged_path = os.path.join(render_video_path, f"{request_id}.mp4")
    return_code = subprocess.call(
        [
            "ffmpeg",
            "-i",
            output_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            merged_path,
        ]
    )
    if return_code != 0:
        raise Exception("Merge audio and video failed")

    # 删除原始视频文件
    os.remove(output_path)

    return merged_path


def extract_audio_track(video_path: str) -> str:
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    command = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar %d %s" % (
        video_path,
        config.extract_audio_track_sample_rate,
        audio_path,
    )
    command_code = subprocess.call(command, shell=True, stdout=None)
    if command_code != 0:
        raise Exception("提取音频失败")
    return audio_path
