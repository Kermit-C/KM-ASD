#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 结果输出
@Author: Kermit
@Date: 2024-02-18 15:55:11
"""


from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.reduce_message_body import ReduceMessageBody
from event_bus.message_body.result_message_body import ResultMessageBody


class ReduceProcessor(BaseEventBusProcessor):
    """结果输出处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)

    def process(self, event_message_body: ReduceMessageBody):
        if event_message_body.type == "ASD":
            self._process_asd(event_message_body)
        elif event_message_body.type == "FACE_RECOGNIZE":
            self._process_face_recognize(event_message_body)
        elif event_message_body.type == "SPEAKER_VERIFICATE":
            self._process_speaker_verificate(event_message_body)
        else:
            raise ValueError(f"Unknown output message type: {event_message_body.type}")

    def _process_asd(self, event_message_body: ReduceMessageBody):
        frame_count: int = event_message_body.frame_count  # type: ignore
        frame_timestamp: int = event_message_body.frame_timestamp  # type: ignore
        frame_face_idx: int = event_message_body.frame_face_idx  # type: ignore
        frame_face_bbox: tuple[int] = event_message_body.frame_face_bbox  # type: ignore
        frame_asd_status: int = event_message_body.frame_asd_status  # type: ignore

        # TODO
        self.result(
            ResultMessageBody(
                frame_count=frame_count,
                frame_timestamp=frame_timestamp,
                frame=event_message_body.frame,  # TODO: 从 extract 的 store 中取
                video_fps=30,  # TODO: 从 extract 的 store 中取
                video_frame_count=100,  # TODO: 从 extract 的 store 中取
                speaker_face_bbox=frame_face_bbox,  # TODO: reduce
                speaker_face_label=frame_face_idx,  # TODO: reduce
                speaker_offscreen_voice_label=frame_asd_status,  # TODO: reduce
            )
        )

    def _process_face_recognize(self, event_message_body: ReduceMessageBody):
        frame_count: int = event_message_body.frame_count  # type: ignore
        frame_timestamp: int = event_message_body.frame_timestamp  # type: ignore
        frame_face_idx: int = event_message_body.frame_face_idx  # type: ignore
        frame_face_label: str = event_message_body.frame_face_label  # type: ignore
        # TODO

    def _process_speaker_verificate(self, event_message_body: ReduceMessageBody):
        audio_sample_rate: int = event_message_body.audio_sample_rate  # type: ignore
        audio_frame_length: int = event_message_body.audio_frame_length  # type: ignore
        audio_frame_step: int = event_message_body.audio_frame_step  # type: ignore
        audio_frame_count: int = event_message_body.audio_frame_count  # type: ignore
        audio_frame_timestamp: int = event_message_body.audio_frame_timestamp  # type: ignore
        frame_voice_label: str = event_message_body.frame_voice_label  # type: ignore
        # TODO
