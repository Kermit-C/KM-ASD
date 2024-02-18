#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 说话人验证处理器
@Author: Kermit
@Date: 2024-02-18 15:58:15
"""


from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.reduce_message_body import ReduceMessageBody
from event_bus.message_body.speaker_verificate_message_body import (
    SpeakerVerificateMessageBody,
)
from service.event_bus_service import call_speaker_verification


class SpeakerVerificateProcessor(BaseEventBusProcessor):
    """说话人验证处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: SpeakerVerificateMessageBody):
        # TODO: 考虑聚合多个音频帧再进行验证
        label = call_speaker_verification(event_message_body.audio_pcm)
        self.publish_next(
            "reduce_topic",
            ReduceMessageBody(
                type="SPEAKER_VERIFICATE",
                audio_sample_rate=event_message_body.audio_sample_rate,
                audio_frame_length=event_message_body.audio_frame_length,
                audio_frame_step=event_message_body.audio_frame_step,
                audio_frame_count=event_message_body.audio_frame_count,
                audio_frame_timestamp=event_message_body.audio_frame_timestamp,
                frame_voice_label=label,
            ),
        )
