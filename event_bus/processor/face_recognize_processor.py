#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸识别处理器
@Author: Kermit
@Date: 2024-02-18 15:51:15
"""

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.face_recognize_message_body import FaceRecognizeMessageBody
from event_bus.message_body.reduce_message_body import ReduceMessageBody
from service.event_bus_service import call_face_recognition


class FaceRecognizeProcessor(BaseEventBusProcessor):
    """人脸识别处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: FaceRecognizeMessageBody):
        # TODO: 考虑使用前面帧的结果来发消息，节约计算资源
        label = call_face_recognition(
            event_message_body.frame, event_message_body.face_lmks
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
