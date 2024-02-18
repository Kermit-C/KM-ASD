#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸裁剪处理器
@Author: Kermit
@Date: 2024-02-18 15:51:15
"""

import numpy as np

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.asd_message_body import AsdMessageBody
from event_bus.message_body.face_crop_message_body import FaceCropMessageBody
from event_bus.message_body.face_recognize_message_body import FaceRecognizeMessageBody


class FaceCropProcessor(BaseEventBusProcessor):
    """人脸裁剪处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: FaceCropMessageBody):
        frame = event_message_body.frame
        face_dets = event_message_body.face_dets

        for face_idx, face_det in enumerate(face_dets):
            x1, y1, x2, y2 = face_det["bbox"]
            left_eye_x = face_det["landmarks"]["left_eye"]["x"]
            left_eye_y = face_det["landmarks"]["left_eye"]["y"]
            right_eye_x = face_det["landmarks"]["right_eye"]["x"]
            right_eye_y = face_det["landmarks"]["right_eye"]["y"]
            nose_x = face_det["landmarks"]["nose"]["x"]
            nose_y = face_det["landmarks"]["nose"]["y"]
            left_mouth_x = face_det["landmarks"]["left_mouth"]["x"]
            left_mouth_y = face_det["landmarks"]["left_mouth"]["y"]
            right_mouth_x = face_det["landmarks"]["right_mouth"]["x"]
            right_mouth_y = face_det["landmarks"]["right_mouth"]["y"]

            face = frame[y1:y2, x1:x2]
            left_eye_x -= x1
            left_eye_y -= y1
            right_eye_x -= x1
            right_eye_y -= y1
            nose_x -= x1
            nose_y -= y1
            left_mouth_x -= x1
            left_mouth_y -= y1
            right_mouth_x -= x1
            right_mouth_y -= y1

            self.publish_next(
                "asd_topic",
                AsdMessageBody(
                    type="V",
                    frame_count=event_message_body.frame_count,
                    frame_timestamp=event_message_body.frame_timestamp,
                    frame=face,
                    frame_face_idx=face_idx,
                    frame_face_bbox=(x1, y1, x2, y2),  # type: ignore
                ),
            )
            self.publish_next(
                "face_recognize_topic",
                FaceRecognizeMessageBody(
                    event_message_body.frame_count,
                    event_message_body.frame_timestamp,
                    face,
                    face_idx,
                    np.array(
                        [
                            [left_eye_x, left_eye_y],
                            [right_eye_x, right_eye_y],
                            [nose_x, nose_y],
                            [left_mouth_x, left_mouth_y],
                            [right_mouth_x, right_mouth_y],
                        ]
                    ).astype(np.float32),
                ),
            )
