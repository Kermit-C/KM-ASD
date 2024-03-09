#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 结果输出
@Author: Kermit
@Date: 2024-02-18 15:55:11
"""


from threading import RLock
from typing import Optional

import cv2
import numpy as np

import config
from store.local_store import LocalStore

from ..event_bus_factory import get_processor
from ..event_bus_processor import BaseEventBusProcessor
from ..message_body.reduce_message_body import ReduceMessageBody
from ..message_body.result_message_body import ResultMessageBody
from ..processor.audio_to_pcm_processor import AudioToPcmProcessor
from ..store.audio_to_pcm_store import AudioToPcmStore
from ..store.reduce_store import ReduceStore
from ..store.video_to_frame_store import VideoToFrameStore
from .video_to_frame_processor import VideoToFrameProcessor


class ReduceProcessor(BaseEventBusProcessor):
    """结果输出处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)
        self.store = ReduceStore(LocalStore.create)
        self.result_lock = RLock()

    def process(self, event_message_body: ReduceMessageBody):
        if event_message_body.type == "ASD":
            self._process_asd(event_message_body)
        elif event_message_body.type == "FACE_RECOGNIZE":
            self._process_face_recognize(event_message_body)
        elif event_message_body.type == "SPEAKER_VERIFICATE":
            self._process_speaker_verificate(event_message_body)
        else:
            raise ValueError(f"Unknown output message type: {event_message_body.type}")
        self._process_result(event_message_body)

    def process_exception(
        self, event_message_body: ReduceMessageBody, exception: Exception
    ):
        raise Exception("ReduceProcessor process_exception", exception)

    def _process_asd(self, event_message_body: ReduceMessageBody):
        frame_count = event_message_body.frame_count
        frame_face_count = event_message_body.frame_face_count
        frame_timestamp = event_message_body.frame_timestamp
        assert frame_count is not None
        assert frame_face_count is not None
        assert frame_timestamp is not None
        frame_face_idx = event_message_body.frame_face_idx
        frame_face_bbox = event_message_body.frame_face_bbox
        frame_asd_status = event_message_body.frame_asd_status

        self.store.save_frame_result(
            self.get_request_id(),
            frame_count,
            frame_timestamp,
            frame_face_idx=frame_face_idx,
            frame_face_count=frame_face_count,
            frame_face_bbox=frame_face_bbox,
            frame_face_asd_status=frame_asd_status,
        )

    def _process_face_recognize(self, event_message_body: ReduceMessageBody):
        frame_count: int = event_message_body.frame_count  # type: ignore
        frame_timestamp: int = event_message_body.frame_timestamp  # type: ignore
        frame_face_idx: int = event_message_body.frame_face_idx  # type: ignore
        frame_face_label: str = event_message_body.frame_face_label  # type: ignore

        self.store.save_frame_result(
            self.get_request_id(),
            frame_count,
            frame_timestamp,
            frame_face_idx=frame_face_idx,
            frame_face_label=frame_face_label,
        )

    def _process_speaker_verificate(self, event_message_body: ReduceMessageBody):
        audio_sample_rate: int = event_message_body.audio_sample_rate  # type: ignore
        audio_frame_length: int = event_message_body.audio_frame_length  # type: ignore
        audio_frame_step: int = event_message_body.audio_frame_step  # type: ignore
        audio_frame_count: int = event_message_body.audio_frame_count  # type: ignore
        audio_frame_timestamp: int = event_message_body.audio_frame_timestamp  # type: ignore
        frame_voice_label: str = event_message_body.frame_voice_label  # type: ignore

        frame_count, frame_timestamp = (
            self._get_video_frame_count_timestamp_from_near_timestamp(
                self.get_request_id(), audio_frame_timestamp
            )
        )
        if frame_count is None or frame_timestamp is None:
            return

        self.store.save_frame_result(
            self.get_request_id(),
            frame_count,
            frame_timestamp,
            frame_voice_label=frame_voice_label,
        )

    def _process_result(self, event_message_body: ReduceMessageBody):
        if event_message_body.type == "SPEAKER_VERIFICATE":
            return
        # 保证结果输出的顺序
        with self.result_lock:
            assert event_message_body.frame_count is not None
            assert event_message_body.frame_timestamp is not None
            is_frame_result_complete = self.store.is_frame_result_complete(
                self.get_request_id(), event_message_body.frame_count
            )
            is_frame_resulted = self.store.is_frame_resulted(
                self.get_request_id(), event_message_body.frame_count
            )
            is_frame_before_all_resulted = self.store.is_frame_before_all_resulted(
                self.get_request_id(), event_message_body.frame_count
            )
            if (
                not is_frame_result_complete
                or is_frame_resulted
                or not is_frame_before_all_resulted
            ):
                # 未收集到所有结果，或者已经输出过，或之前的帧还没输出，不输出
                # TODO: 实现实时后，需要考虑超时，超时不管是否收集到所有结果都输出
                return

            frame_count = event_message_body.frame_count
            while True:
                wait_result_frame_count_list: list[int] = [frame_count]
                wait_result_frame_count_list += (
                    self.store.get_frame_after_all_complete_but_not_resulted(
                        self.get_request_id(), frame_count
                    )
                )
                for frame_count in wait_result_frame_count_list:
                    self._process_frame_result(frame_count)

                # 判断一下下一个帧是否已经完整，以防处理上面的时候有新的帧进来，导致卡死
                frame_count = wait_result_frame_count_list[-1] + 1
                if not self.store.is_frame_result_complete(
                    self.get_request_id(), frame_count
                ):
                    break

    def _process_frame_result(self, frame_count: int):
        frame_result = self.store.get_frame_result(self.get_request_id(), frame_count)
        frame_timestamp = frame_result["frame_timestamp"]

        speaker_face_bbox = []
        speaker_face_label = []
        speaker_offscreen_voice_label: list[str] = frame_result["frame_voice_label"]
        non_speaker_face_bbox = []
        non_speaker_face_label = []
        for i in range(frame_result["frame_face_count"]):
            if frame_result["frame_face_asd_status"][i] == 1:
                speaker_face_bbox.append(frame_result["frame_face_bbox"][i])
                speaker_face_label.append(frame_result["frame_face_label"][i])
            else:
                non_speaker_face_bbox.append(frame_result["frame_face_bbox"][i])
                non_speaker_face_label.append(frame_result["frame_face_label"][i])

        frame = self._get_video_frame_from_timestamp(
            self.get_request_id(), frame_timestamp
        )
        if frame is None:
            return
        frame = self._render_frame(
            frame.copy(),
            speaker_face_bbox,
            speaker_face_label,
            speaker_offscreen_voice_label,
            non_speaker_face_bbox,
            non_speaker_face_label,
        )

        info = self._get_video_info(self.get_request_id())

        self.result(
            ResultMessageBody(
                frame_count=frame_count,
                frame_timestamp=frame_timestamp,
                frame=frame,
                video_fps=info["video_fps"],
                video_frame_count=info["video_frame_count"],
                speaker_face_bbox=speaker_face_bbox,
                speaker_face_label=speaker_face_label,
                speaker_offscreen_voice_label=speaker_offscreen_voice_label,
                non_speaker_face_bbox=non_speaker_face_bbox,
                non_speaker_face_label=non_speaker_face_label,
            )
        )
        self.store.set_frame_resulted(self.get_request_id(), frame_count)

    def _get_video_frame_from_timestamp(
        self, request_id: str, timestamp: int
    ) -> Optional[np.ndarray]:
        """获取视频帧图像"""
        video_to_frame_store = self._get_video_to_frame_store()
        return video_to_frame_store.get_frame_from_timestamp(request_id, timestamp)

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

    def _get_video_info(self, request_id: str) -> dict:
        """获取视频信息"""
        video_to_frame_store = self._get_video_to_frame_store()
        return video_to_frame_store.get_info(request_id)

    def _get_video_to_frame_store(self) -> VideoToFrameStore:
        """获取视频帧存储器"""
        video_to_frame_processor = get_processor(
            config.event_bus["processors"]["VideoToFrameProcessor"]["processor_name"]
        )
        assert isinstance(video_to_frame_processor, VideoToFrameProcessor)
        return video_to_frame_processor.store

    def _get_audio_to_pcm_store(self) -> AudioToPcmStore:
        """获取音频存储器"""
        audio_to_pcm_store = get_processor(
            config.event_bus["processors"]["AudioToPcmProcessor"]["processor_name"]
        )
        assert isinstance(audio_to_pcm_store, AudioToPcmProcessor)
        return audio_to_pcm_store.store

    def _render_frame(
        self,
        frame: np.ndarray,
        speaker_face_bbox: list[tuple[int, int, int, int]],
        speaker_face_label: list[str],
        speaker_offscreen_voice_label: list[str],
        non_speaker_face_bbox: list[tuple[int, int, int, int]],
        non_speaker_face_label: list[str],
    ):
        # 给 frame 画框，说话人绿色，非说话人红色
        for bbox in speaker_face_bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        for bbox in non_speaker_face_bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        # 给 frame 写标签，说话人绿色，非说话人红色
        for i, bbox in enumerate(speaker_face_bbox):
            cv2.putText(
                frame,
                speaker_face_label[i],
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        for i, bbox in enumerate(non_speaker_face_bbox):
            cv2.putText(
                frame,
                non_speaker_face_label[i],
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        # 给 frame 下方写声音标签
        if len(speaker_offscreen_voice_label) > 0:
            cv2.putText(
                frame,
                "Offscreen speaker: " + ", ".join(speaker_offscreen_voice_label),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        return frame
