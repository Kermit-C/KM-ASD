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
            # TODO: 根据前一 lag 平移移位置

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
