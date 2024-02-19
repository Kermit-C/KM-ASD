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
from event_bus.store.face_crop_store import FaceCropStore
from store.local_store import LocalStore


class FaceCropProcessor(BaseEventBusProcessor):
    """人脸裁剪处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)
        self.store = FaceCropStore(LocalStore.create)
        self.same_face_between_frames_iou_threshold = 0.5

    def process(self, event_message_body: FaceCropMessageBody):
        frame = event_message_body.frame
        face_dets = event_message_body.face_dets

        # TODO: 上一帧还没处理完的情况
        last_frame_faces = self.store.get_faces(
            self.get_request_id(), event_message_body.frame_count - 1
        )

        # 可用的人脸索引
        face_idx_list = list(range(len(face_dets)))
        # 已经分配的人脸索引
        face_idx_alloc_list = []
        for face_det in face_dets:
            x1, y1, x2, y2 = face_det["bbox"]
            face_bbox: tuple[int, int, int, int] = (x1, y1, x2, y2)  # type: ignore

            # 从上一帧人脸中找到最接近的人脸
            face_idx = self.get_face_idx_from_last_frame(last_frame_faces, face_bbox)
            if face_idx is not None:
                face_idx_list = list(filter(lambda x: x != face_idx, face_idx_list))
                if face_idx in face_idx_alloc_list:
                    # 如果已经分配了这个人脸，就把挑出来使用可用的人脸索引
                    alloc_idx = face_idx_alloc_list.index(face_idx)
                    face_idx_alloc_list[alloc_idx] = face_idx_list.pop(0)
            else:
                # 如果没有找到最接近的人脸，就使用可用的人脸索引
                face_idx = face_idx_list.pop(0)

            face_idx_alloc_list.append(face_idx)

        for face_idx, face_det in zip(face_idx_alloc_list, face_dets):
            x1, y1, x2, y2 = face_det["bbox"]
            face_bbox: tuple[int, int, int, int] = (x1, y1, x2, y2)  # type: ignore

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

            self.store.save_face(
                self.get_request_id(),
                event_message_body.frame_count,
                event_message_body.frame_timestamp,
                face,
                face_idx,
                face_bbox,
            )
            self.publish_next(
                "asd_topic",
                AsdMessageBody(
                    type="V",
                    frame_count=event_message_body.frame_count,
                    frame_timestamp=event_message_body.frame_timestamp,
                    frame=face,
                    frame_face_idx=face_idx,
                    frame_face_count=len(face_dets),
                    frame_face_bbox=face_bbox,
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

    def get_face_idx_from_last_frame(
        self, last_frame_faces: list[dict], face_bbox: tuple[int, int, int, int]
    ):
        """从上一帧人脸中找到最接近的人脸"""
        for last_frame_face in last_frame_faces:
            last_frame_face_bbox = last_frame_face["frame_face_bbox"]
            # 计算两个人脸框的IOU
            iou = self.get_iou(face_bbox, last_frame_face_bbox)
            if iou > self.same_face_between_frames_iou_threshold:
                return last_frame_face["frame_face_idx"]
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
