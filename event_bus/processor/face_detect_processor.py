#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸检测处理器
@Author: Kermit
@Date: 2024-02-18 15:51:15
"""

import asyncio
import logging

from event_bus.event_bus_excecutor import get_event_loop
from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.face_crop_message_body import FaceCropMessageBody
from event_bus.message_body.face_detect_message_body import FaceDetectMessageBody
from event_bus.store.face_detect_store import FaceDetectStore
from service.event_bus_service import call_face_detection
from store.local_store import LocalStore


class FaceDetectProcessor(BaseEventBusProcessor):
    """人脸检测处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name, is_async=True)
        self.store = FaceDetectStore(LocalStore.create)
        self.detect_lag: int = self.processor_properties["detect_lag"]
        self.same_face_between_frames_iou_threshold: float = self.processor_properties[
            "same_face_between_frames_iou_threshold"
        ]

    async def process_async(self, event_message_body: FaceDetectMessageBody):
        lag_idx = (event_message_body.frame_count - 1) % self.detect_lag
        if lag_idx == 0:
            face_dets = await call_face_detection(
                event_message_body.frame, self.processor_timeout
            )
        else:
            face_dets = None
            wait_time = 0
            while face_dets is None:
                # 取这一个 lag 的第一个帧的人脸
                face_dets = self.store.get_faces(
                    self.get_request_id(), event_message_body.frame_count - lag_idx
                )
                if face_dets is None:
                    await asyncio.sleep(0.1)
                    wait_time += 0.1
                    if wait_time > self.processor_timeout:
                        raise Exception("face detect timeout")

            # 根据前一 lag 平移移位置
            last_lag_face_dets = self.store.get_faces(
                self.get_request_id(),
                event_message_body.frame_count - lag_idx - self.detect_lag,
            )
            if last_lag_face_dets is not None:
                for i, face_det in enumerate(face_dets):
                    last_lag_face_det = self.get_face_det_from_last_lag(
                        last_lag_face_dets, face_det
                    )
                    if last_lag_face_det is not None:
                        face_dets[i] = self.move_face_det_by_last_lag(
                            lag_idx / self.detect_lag, face_det, last_lag_face_det
                        )

        await self.store.save_face(
            self.get_request_id(),
            event_message_body.frame_count,
            event_message_body.frame_timestamp,
            face_dets,
        )
        self.publish_next(
            "face_crop_topic",
            FaceCropMessageBody(
                event_message_body.frame_count,
                event_message_body.frame_timestamp,
                event_message_body.frame,
                face_dets,
            ),
        )

    async def process_exception_async(
        self, event_message_body: FaceCropMessageBody, exception: Exception
    ):
        await self.store.save_face(
            self.get_request_id(),
            event_message_body.frame_count,
            event_message_body.frame_timestamp,
            [],
        )
        self.publish_next(
            "face_crop_topic",
            FaceCropMessageBody(
                event_message_body.frame_count,
                event_message_body.frame_timestamp,
                event_message_body.frame,
                [],
            ),
        )

    def get_face_det_from_last_lag(
        self, last_frame_face_dets: list[dict], face_det: dict
    ):
        """从上一 lag 人脸中找到最接近的人脸"""
        face_bbox = (
            face_det["bbox"]["x1"],
            face_det["bbox"]["y1"],
            face_det["bbox"]["x2"],
            face_det["bbox"]["y2"],
        )
        for last_frame_face_det in last_frame_face_dets:
            last_frame_face_bbox = (
                last_frame_face_det["bbox"]["x1"],
                last_frame_face_det["bbox"]["y1"],
                last_frame_face_det["bbox"]["x2"],
                last_frame_face_det["bbox"]["y2"],
            )
            # 计算两个人脸框的IOU
            iou = self.get_iou(face_bbox, last_frame_face_bbox)
            if iou > self.same_face_between_frames_iou_threshold:
                return last_frame_face_det
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

    def move_face_det_by_last_lag(self, lag_rate, curr_lag_face_det, last_lag_face_det):
        """根据上一 lag 的人脸位置，平移当前 lag 的人脸位置"""

        lag_rate = lag_rate * 0.25

        x1 = int(
            lag_rate
            * (curr_lag_face_det["bbox"]["x1"] - last_lag_face_det["bbox"]["x1"])
            + curr_lag_face_det["bbox"]["x1"]
        )
        y1 = int(
            lag_rate
            * (curr_lag_face_det["bbox"]["y1"] - last_lag_face_det["bbox"]["y1"])
            + curr_lag_face_det["bbox"]["y1"]
        )
        x2 = int(
            lag_rate
            * (curr_lag_face_det["bbox"]["x2"] - last_lag_face_det["bbox"]["x2"])
            + curr_lag_face_det["bbox"]["x2"]
        )
        y2 = int(
            lag_rate
            * (curr_lag_face_det["bbox"]["y2"] - last_lag_face_det["bbox"]["y2"])
            + curr_lag_face_det["bbox"]["y2"]
        )
        width = x2 - x1
        height = y2 - y1
        left_eye_x = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["left_eye"]["x"]
                - last_lag_face_det["landmarks"]["left_eye"]["x"]
            )
            + curr_lag_face_det["landmarks"]["left_eye"]["x"]
        )
        left_eye_y = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["left_eye"]["y"]
                - last_lag_face_det["landmarks"]["left_eye"]["y"]
            )
            + curr_lag_face_det["landmarks"]["left_eye"]["y"]
        )
        right_eye_x = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["right_eye"]["x"]
                - last_lag_face_det["landmarks"]["right_eye"]["x"]
            )
            + curr_lag_face_det["landmarks"]["right_eye"]["x"]
        )
        right_eye_y = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["right_eye"]["y"]
                - last_lag_face_det["landmarks"]["right_eye"]["y"]
            )
            + curr_lag_face_det["landmarks"]["right_eye"]["y"]
        )
        nose_x = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["nose"]["x"]
                - last_lag_face_det["landmarks"]["nose"]["x"]
            )
            + curr_lag_face_det["landmarks"]["nose"]["x"]
        )
        nose_y = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["nose"]["y"]
                - last_lag_face_det["landmarks"]["nose"]["y"]
            )
            + curr_lag_face_det["landmarks"]["nose"]["y"]
        )
        left_mouth_x = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["left_mouth"]["x"]
                - last_lag_face_det["landmarks"]["left_mouth"]["x"]
            )
            + curr_lag_face_det["landmarks"]["left_mouth"]["x"]
        )
        left_mouth_y = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["left_mouth"]["y"]
                - last_lag_face_det["landmarks"]["left_mouth"]["y"]
            )
            + curr_lag_face_det["landmarks"]["left_mouth"]["y"]
        )
        right_mouth_x = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["right_mouth"]["x"]
                - last_lag_face_det["landmarks"]["right_mouth"]["x"]
            )
            + curr_lag_face_det["landmarks"]["right_mouth"]["x"]
        )
        right_mouth_y = int(
            lag_rate
            * (
                curr_lag_face_det["landmarks"]["right_mouth"]["y"]
                - last_lag_face_det["landmarks"]["right_mouth"]["y"]
            )
            + curr_lag_face_det["landmarks"]["right_mouth"]["y"]
        )

        return {
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
            },
            "confidence": curr_lag_face_det["confidence"],
            "landmarks": {
                "left_eye": {
                    "x": left_eye_x,
                    "y": left_eye_y,
                },
                "right_eye": {
                    "x": right_eye_x,
                    "y": right_eye_y,
                },
                "nose": {
                    "x": nose_x,
                    "y": nose_y,
                },
                "left_mouth": {
                    "x": left_mouth_x,
                    "y": left_mouth_y,
                },
                "right_mouth": {
                    "x": right_mouth_x,
                    "y": right_mouth_y,
                },
            },
        }
