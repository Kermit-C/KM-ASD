#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 结果消息体
@Author: Kermit
@Date: 2024-02-18 15:51:15
"""

import numpy as np

from event_bus.event_message import EventMessageBody


class ResultMessageBody(EventMessageBody):
    """结果消息体"""

    def __init__(
        self,
        frame_count: int,
        frame_timestamp: int,
        frame: np.ndarray,
        video_fps: float,
        video_frame_count: int,
        speaker_face_bbox: list[tuple[int, int, int, int]],
        speaker_face_label: list[str],
        speaker_offscreen_voice_label: list[str],
        non_speaker_face_bbox: list[tuple[int, int, int, int]],
        non_speaker_face_label: list[str],
    ):
        # 帧数，代表第几帧
        self.frame_count = frame_count
        # 帧时间戳
        self.frame_timestamp = frame_timestamp
        # 帧图像
        self.frame = frame
        # 视频帧率
        self.video_fps = video_fps
        # 视频帧数
        self.video_frame_count = video_frame_count
        # 说话人人脸框
        self.speaker_face_bbox = speaker_face_bbox
        # 说话人人脸标签
        self.speaker_face_label = speaker_face_label
        # 画面外说话人声音标签
        self.speaker_voice_label = speaker_offscreen_voice_label
        # 未说话人人脸框
        self.non_speaker_face_bbox = non_speaker_face_bbox
        # 未说话人人脸标签
        self.non_speaker_face_label = non_speaker_face_label
