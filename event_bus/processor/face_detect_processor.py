#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸检测处理器
@Author: Kermit
@Date: 2024-02-18 15:51:15
"""

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.face_crop_message_body import FaceCropMessageBody
from event_bus.message_body.face_detect_message_body import FaceDetectMessageBody
from service.event_bus_service import call_face_detection


class FaceDetectProcessor(BaseEventBusProcessor):
    """人脸检测处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: FaceDetectMessageBody):
        face_dets = call_face_detection(event_message_body.frame)
        self.publish_next(
            "face_crop_topic",
            FaceCropMessageBody(
                event_message_body.frame_count,
                event_message_body.frame_timestamp,
                event_message_body.frame,
                face_dets,
            ),
        )
