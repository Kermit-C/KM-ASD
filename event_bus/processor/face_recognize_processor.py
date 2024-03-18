#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸识别处理器
@Author: Kermit
@Date: 2024-02-18 15:51:15
"""

import logging

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.face_recognize_message_body import FaceRecognizeMessageBody
from event_bus.message_body.reduce_message_body import ReduceMessageBody
from event_bus.store.face_recognize_store import FaceRecognizeStore
from service.event_bus_service import call_face_recognition
from store.local_store import LocalStore


class FaceRecognizeProcessor(BaseEventBusProcessor):
    """人脸识别处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name, is_async=True)
        self.store = FaceRecognizeStore(LocalStore.create)
        self.same_face_between_frames_iou_threshold: float = self.processor_properties[
            "same_face_between_frames_iou_threshold"
        ]

    async def process_async(self, event_message_body: FaceRecognizeMessageBody):
        # 获取最近 10 帧最接近的人脸的标签
        for frame_count in range(
            event_message_body.frame_count - 1, event_message_body.frame_count - 10, -1
        ):
            # 获取上一帧的人脸
            last_frame_faces = self.store.get_faces(self.get_request_id(), frame_count)
            if last_frame_faces is None:
                last_frame_faces = []
            # 从上一帧人脸中找到最接近的人脸的标签
            last_label = self.get_face_label_from_last_frame(
                last_frame_faces, event_message_body.frame_face_bbox
            )
            if last_label is not None:
                break

        if last_label is not None:
            label = last_label
        else:
            label = await call_face_recognition(
                event_message_body.frame,
                event_message_body.face_lmks,
                self.processor_timeout,
            )
        await self.store.save_face(
            self.get_request_id(),
            event_message_body.frame_count,
            event_message_body.frame_timestamp,
            event_message_body.frame,
            event_message_body.frame_face_idx,
            event_message_body.frame_face_bbox,
            label,
        )
        self.publish_next(
            "reduce_topic",
            ReduceMessageBody(
                type="FACE_RECOGNIZE",
                frame_count=event_message_body.frame_count,
                frame_timestamp=event_message_body.frame_timestamp,
                frame_face_idx=event_message_body.frame_face_idx,
                frame_face_label=label,
            ),
        )

    async def process_exception_async(
        self, event_message_body: FaceRecognizeMessageBody, exception: Exception
    ):
        # logging.error("FaceRecognizeProcessor process_exception", exception)
        self.publish_next(
            "reduce_topic",
            ReduceMessageBody(
                type="FACE_RECOGNIZE",
                frame_count=event_message_body.frame_count,
                frame_timestamp=event_message_body.frame_timestamp,
                frame_face_idx=event_message_body.frame_face_idx,
                frame_face_label="Unknown",
            ),
        )

    def get_face_label_from_last_frame(
        self, last_frame_faces: list[dict], face_bbox: tuple[int, int, int, int]
    ):
        """从上一帧人脸中找到最接近的人脸"""
        for last_frame_face in last_frame_faces:
            last_frame_face_bbox = last_frame_face["frame_face_bbox"]
            # 计算两个人脸框的IOU
            iou = self.get_iou(face_bbox, last_frame_face_bbox)
            if iou > self.same_face_between_frames_iou_threshold:
                return last_frame_face["frame_face_label"]
        return None

    def get_iou(
        self, box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]
    ) -> float:
        """计算两个人脸框的IOU"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        x_overlap = max(0, min(x2, x4) - max(x1, x3))
        y_overlap = max(0, min(y2, y4) - max(y1, y3))
        intersection = x_overlap * y_overlap
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        return intersection / union
