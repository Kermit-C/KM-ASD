#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 视频转图片帧处理器
@Author: Kermit
@Date: 2024-02-18 15:12:20
"""

from typing import Generator

import cv2
import numpy as np

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.face_detect_message_body import FaceDetectMessageBody
from event_bus.message_body.video_to_frame_message_body import VideoToFrameMessageBody
from event_bus.store.video_to_frame_store import VideoToFrameStore
from store.local_store import LocalStore


class VideoToFrameProcessor(BaseEventBusProcessor):
    """视频转图片帧处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)
        self.store = VideoToFrameStore(LocalStore.create)

    def _capture(
        self, video_path: str
    ) -> Generator[tuple[np.ndarray, int, int, float, int], None, None]:
        # 打开视频流
        video_capture = cv2.VideoCapture(video_path)
        video_fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # 用于计数帧数
        frame_count = 0
        while True:
            # 读取的 frame 是 BGR 格式
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_count += 1
            frame_timestamp = int((frame_count / video_fps) * 1000)
            yield frame, frame_count, frame_timestamp, video_fps, video_frame_count

        # 释放资源
        video_capture.release()

    def process(self, event_message_body: VideoToFrameMessageBody):
        skipped_frame = 0
        for (
            frame,
            frame_count,
            frame_timestamp,
            source_video_fps,
            video_frame_count,
        ) in self._capture(event_message_body.video_path):
            # 按帧率缩放
            self.processor_properties["target_video_fps"] = float(
                self.processor_properties["target_video_fps"]
            )
            if source_video_fps / self.processor_properties["target_video_fps"] > 1:
                # 将过大的帧率转成目标帧率
                if (
                    frame_count
                    % int(
                        source_video_fps / self.processor_properties["target_video_fps"]
                    )
                    != 0
                ):
                    skipped_frame += 1
                    continue

            frame_count -= skipped_frame
            video_fps = (
                source_video_fps
                if skipped_frame == 0
                else self.processor_properties["target_video_fps"]
            )

            self.store.save_frame(
                self.get_request_id(), frame_count, frame_timestamp, frame
            )
            self.store.save_info(
                self.get_request_id(),
                video_fps,
                video_frame_count,
            )
            # TODO: 最终需要去掉 or True
            if self.is_real_time() or True:
                # 实时处理
                # 等待人脸检测结果，最大等待是帧间隔
                self.publish_next(
                    "face_detect_topic",
                    FaceDetectMessageBody(frame_count, frame_timestamp, frame),
                    is_async=True,
                    is_wait_async=True,
                    wait_async_timeout=(1 / video_fps),
                )
            else:
                # 非实时处理
                # 同步等待
                self.publish_next(
                    "face_detect_topic",
                    FaceDetectMessageBody(frame_count, frame_timestamp, frame),
                    is_async=False,
                )

    def process_exception(
        self, event_message_body: VideoToFrameMessageBody, exception: Exception
    ):
        raise Exception("VideoToFrameProcessor process_exception", exception)
