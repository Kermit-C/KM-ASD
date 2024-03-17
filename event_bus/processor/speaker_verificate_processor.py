#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 说话人验证处理器
@Author: Kermit
@Date: 2024-02-18 15:58:15
"""


import logging

import torch

from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.reduce_message_body import ReduceMessageBody
from event_bus.message_body.speaker_verificate_message_body import (
    SpeakerVerificateMessageBody,
)
from event_bus.store.speaker_verificate_store import SpeakerVerificateStore
from service.event_bus_service import call_speaker_verification
from store.local_store import LocalStore


class SpeakerVerificateProcessor(BaseEventBusProcessor):
    """说话人验证处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)
        self.store = SpeakerVerificateStore(LocalStore.create)
        # 需要聚合成的大帧的采样点长度，每聚合一次，就会调用一次模型
        self.aggregate_frame_length: int = self.processor_properties[
            "aggregate_frame_length"
        ]

    async def process_async(self, event_message_body: SpeakerVerificateMessageBody):
        self.store.save_frame(
            self.get_request_id(),
            event_message_body.audio_frame_count,
            event_message_body.audio_frame_timestamp,
            event_message_body.audio_sample_rate,
            event_message_body.audio_frame_length,
            event_message_body.audio_frame_step,
            event_message_body.audio_pcm,
        )

        frame_number_of_aggregate = self.get_frame_number_of_aggregate(
            self.aggregate_frame_length,
            event_message_body.audio_frame_step,
            event_message_body.audio_frame_length,
        )
        # 如果不是聚合帧的整数倍，则不处理
        if event_message_body.audio_frame_count % frame_number_of_aggregate != 0:
            return

        # 获取聚合帧
        frames: list[torch.Tensor] = []
        for frame_count in range(
            event_message_body.audio_frame_count - frame_number_of_aggregate + 1,
            event_message_body.audio_frame_count + 1,
        ):
            frame = self.store.get_frame(self.get_request_id(), frame_count)
            if frame is None:
                return
            frames.append(frame["audio_frame"])
        aggregate_frame = self.aggregate_frames(
            frames,
            event_message_body.audio_frame_step,
            event_message_body.audio_frame_length,
        )

        label = await call_speaker_verification(aggregate_frame, self.processor_timeout)
        if not label:
            return

        # 所有聚合帧都使用这个标签
        for frame_count in range(
            event_message_body.audio_frame_count - frame_number_of_aggregate + 1,
            event_message_body.audio_frame_count + 1,
        ):
            self.store.save_frame_label(
                self.get_request_id(),
                frame_count,
                label,
            )
            frame = self.store.get_frame(self.get_request_id(), frame_count)
            assert frame is not None
            self.publish_next(
                "reduce_topic",
                ReduceMessageBody(
                    type="SPEAKER_VERIFICATE",
                    audio_sample_rate=event_message_body.audio_sample_rate,
                    audio_frame_length=event_message_body.audio_frame_length,
                    audio_frame_step=event_message_body.audio_frame_step,
                    audio_frame_count=frame_count,
                    audio_frame_timestamp=frame["audio_frame_timestamp"],
                    frame_voice_label=label,
                ),
            )

    def process_exception(
        self, event_message_body: SpeakerVerificateMessageBody, exception: Exception
    ):
        # logging.error("FaceRecognizeProcessor process_exception", exception)
        pass

    def get_frame_number_of_aggregate(
        self, aggregate_frame_length: int, frame_step: int, frame_length: int
    ):
        """获取聚合帧数"""
        return (aggregate_frame_length - frame_length) // frame_step + 1

    def aggregate_frames(
        self, frames: list[torch.Tensor], frame_step: int, frame_length: int
    ):
        """聚合音频帧，排除重叠部分"""
        aggregate_frame_length = (len(frames) - 1) * frame_step + frame_length
        aggregate_frame = torch.zeros(aggregate_frame_length, dtype=torch.float32)
        for i, frame in enumerate(frames):
            start = i * frame_step
            end = start + frame_length
            aggregate_frame[start:end] = frame

        return aggregate_frame
