#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 说话人检测处理器
@Author: Kermit
@Date: 2024-02-18 15:53:31
"""


import logging
from threading import RLock
from typing import Optional

import numpy as np

import config
from event_bus.event_bus_factory import get_processor
from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.asd_message_body import AsdMessageBody
from event_bus.message_body.reduce_message_body import ReduceMessageBody
from event_bus.processor.audio_to_pcm_processor import AudioToPcmProcessor
from event_bus.processor.video_to_frame_processor import VideoToFrameProcessor
from event_bus.store.asd_store import ActiveSpeakerDetectionStore
from event_bus.store.audio_to_pcm_store import AudioToPcmStore
from event_bus.store.video_to_frame_store import VideoToFrameStore
from service.event_bus_service import call_asd
from store.local_store import LocalStore


class AsdProcessor(BaseEventBusProcessor):
    """说话人检测处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)
        self.store = ActiveSpeakerDetectionStore(LocalStore.create)
        self.frames_per_clip: int = self.processor_properties["frmc"]
        self.asd_create_lock_lock: RLock = RLock()
        self.asd_lock_of_request: dict[str, RLock] = {}

    def process(self, event_message_body: AsdMessageBody):
        if event_message_body.type == "V":
            assert event_message_body.frame_count is not None
            assert event_message_body.frame_timestamp is not None
            assert event_message_body.frame is not None
            assert event_message_body.frame_face_idx is not None
            assert event_message_body.frame_face_count is not None
            assert event_message_body.frame_face_bbox is not None
            assert event_message_body.frame_height is not None
            assert event_message_body.frame_width is not None
            self.store.save_frame(
                request_id=self.get_request_id(),
                frame_count=event_message_body.frame_count,
                frame_timestamp=event_message_body.frame_timestamp,
                face_frame=event_message_body.frame,
                face_idx=event_message_body.frame_face_idx,
                face_bbox=event_message_body.frame_face_bbox,
                frame_face_count=event_message_body.frame_face_count,
                frame_height=event_message_body.frame_height,
                frame_width=event_message_body.frame_width,
            )
            self._process_asd(event_message_body.frame_count)
        elif event_message_body.type == "A":
            assert event_message_body.audio_frame_count is not None
            assert event_message_body.audio_frame_timestamp is not None
            assert event_message_body.audio_pcm is not None
            parsed_frame_count, parsed_frame_timestamp = (
                self._get_video_frame_count_timestamp_from_near_timestamp(
                    self.get_request_id(), event_message_body.audio_frame_timestamp
                )
            )
            if parsed_frame_count is None or parsed_frame_timestamp is None:
                return
            self.store.save_audio_frame(
                request_id=self.get_request_id(),
                frame_count=parsed_frame_count,
                frame_timestamp=parsed_frame_timestamp,
                audio_frame=self._parse_audio_frame(
                    event_message_body.audio_pcm.numpy(),
                    event_message_body.audio_frame_timestamp,
                ),
            )
            self._process_asd(parsed_frame_count)
        else:
            raise ValueError("AsdProcessor event_message_body.type error")

    def process_exception(
        self, event_message_body: AsdMessageBody, exception: Exception
    ):
        # 已在 process 内部处理过 ASD 调用异常，其他异常直接抛出，视为不可恢复异常
        raise ValueError("AsdProcessor event_message_body.type error", exception)

    def _process_asd(self, frame_count: int):
        with self.asd_create_lock_lock:
            if self.get_request_id() not in self.asd_lock_of_request:
                self.asd_lock_of_request[self.get_request_id()] = RLock()
                asd_lock_of_request = self.asd_lock_of_request[self.get_request_id()]
            else:
                asd_lock_of_request = self.asd_lock_of_request[self.get_request_id()]

        with asd_lock_of_request:
            if (
                self.store.is_frame_completed(self.get_request_id(), frame_count)
                and not self.store.is_frame_asded(self.get_request_id(), frame_count)
                and self.store.is_frame_before_all_asded(
                    self.get_request_id(), frame_count
                )
            ):
                while True:
                    wait_asd_frame_count_list: list[int] = [
                        frame_count
                    ] + self.store.get_frame_after_all_completed(
                        self.get_request_id(), frame_count
                    )

                    for wait_asd_frame_count in wait_asd_frame_count_list:
                        face_dicts = self.store.get_frame_faces(
                            self.get_request_id(), wait_asd_frame_count
                        )
                        faces: list[np.ndarray] = [
                            face_dict["face_frame"] for face_dict in face_dicts
                        ]
                        face_bboxes: list[tuple[int, int, int, int]] = [
                            face_dict["face_bbox"] for face_dict in face_dicts
                        ]
                        audio: np.ndarray = self.store.get_frame_audio(
                            self.get_request_id(), wait_asd_frame_count
                        )  # type: ignore

                        try:
                            is_active_list = call_asd(
                                self.get_request_id(),
                                wait_asd_frame_count,
                                faces,
                                face_bboxes,
                                audio,
                                self.processor_timeout,
                                self.store.get_frame_info(
                                    self.get_request_id(), wait_asd_frame_count  # type: ignore
                                )["frame_height"],
                                self.store.get_frame_info(
                                    self.get_request_id(), wait_asd_frame_count  # type: ignore
                                )["frame_width"],
                            )
                        except:
                            is_active_list = [False] * len(faces)

                        for face_bbox, is_active, face_dict in zip(
                            face_bboxes, is_active_list, face_dicts
                        ):
                            asd_status = 1 if is_active else 0
                            self.publish_next(
                                "reduce_topic",
                                ReduceMessageBody(
                                    type="ASD",
                                    frame_count=wait_asd_frame_count,
                                    frame_timestamp=self.store.get_frame_info(
                                        self.get_request_id(), wait_asd_frame_count
                                    )[  # type: ignore
                                        "frame_timestamp"
                                    ],
                                    frame_face_idx=face_dict["face_idx"],
                                    frame_face_count=self.store.get_frame_info(
                                        self.get_request_id(), wait_asd_frame_count
                                    )[  # type: ignore
                                        "frame_face_count"
                                    ],
                                    frame_face_bbox=face_bbox,
                                    frame_asd_status=asd_status,
                                ),
                            )
                        self.store.set_frame_asded(
                            self.get_request_id(), wait_asd_frame_count
                        )

                    # 判断一下下一个帧是否已经完整，以防处理上面的时候有新的帧进来，导致卡死
                    frame_count = wait_asd_frame_count_list[-1] + 1
                    if not self.store.is_frame_completed(
                        self.get_request_id(), frame_count
                    ):
                        break

        with self.asd_create_lock_lock:
            if self.get_request_id() in self.asd_lock_of_request:
                del self.asd_lock_of_request[self.get_request_id()]

    def _parse_audio_frame(
        self, audio_frame: np.ndarray, audio_frame_timestamp: int
    ) -> np.ndarray:
        """把音频切割到一视频帧的长度，末尾点是音频最后"""
        _, parsed_frame_timestamp = (
            self._get_video_frame_count_timestamp_from_near_timestamp(
                self.get_request_id(), audio_frame_timestamp
            )
        )
        assert parsed_frame_timestamp is not None
        audio_to_pcm_sample_rate, audio_frame_length, audio_frame_step = (
            self._get_audio_to_pcm_config()
        )
        video_fps: float = self._get_video_fps(self.get_request_id())

        lag_timestamp = audio_frame_timestamp - parsed_frame_timestamp
        if lag_timestamp < 0:
            lag_timestamp += int(1000 / video_fps)

        return audio_frame[
            int(
                -(audio_to_pcm_sample_rate / video_fps)
                - (lag_timestamp * audio_to_pcm_sample_rate) / 1000
            ) : int(-(lag_timestamp * audio_to_pcm_sample_rate) / 1000)
        ]

    def _get_video_frame_count_timestamp_from_near_timestamp(
        self, request_id: str, timestamp: int
    ) -> tuple[Optional[int], Optional[int]]:
        """根据时间戳获取最近的视频帧序号和时间戳"""
        video_to_frame_store = self._get_video_to_frame_store()
        frame_timestamp = video_to_frame_store.get_frame_timestamp_from_near_timestamp(
            request_id, timestamp
        )
        if frame_timestamp is None:
            return None, None
        frame_count = video_to_frame_store.get_frame_count_from_timestamp(
            request_id, frame_timestamp
        )
        return frame_count, frame_timestamp

    def _get_video_fps(self, request_id: str) -> float:
        """获取视频帧率"""
        video_to_frame_store = self._get_video_to_frame_store()
        return video_to_frame_store.get_info(request_id)["video_fps"]

    def _get_video_to_frame_store(self) -> VideoToFrameStore:
        """获取视频帧存储器"""
        video_to_frame_store = get_processor(
            config.event_bus["processors"]["VideoToFrameProcessor"]["processor_name"]
        )
        assert isinstance(video_to_frame_store, VideoToFrameProcessor)
        return video_to_frame_store.store

    def _get_audio_to_pcm_config(self) -> tuple[int, int, int]:
        """获取音频转 PCM 处理器配置"""
        audio_to_pcm_processor = get_processor(
            config.event_bus["processors"]["AudioToPcmProcessor"]["processor_name"]
        )
        assert isinstance(audio_to_pcm_processor, AudioToPcmProcessor)
        audio_to_pcm_sample_rate: int = audio_to_pcm_processor.audio_to_pcm_sample_rate
        audio_frame_length: int = audio_to_pcm_processor.frame_length
        audio_frame_step: int = audio_to_pcm_processor.frame_step
        return audio_to_pcm_sample_rate, audio_frame_length, audio_frame_step
