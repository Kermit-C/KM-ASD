#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 音频转 PCM 帧处理器
@Author: Kermit
@Date: 2024-02-18 16:30:53
"""


import time
from typing import Generator, Optional

import torch
import torchaudio

import config
from event_bus.event_bus_factory import get_processor
from event_bus.event_bus_processor import BaseEventBusProcessor
from event_bus.message_body.asd_message_body import AsdMessageBody
from event_bus.message_body.audio_to_pcm_message_body import AudioToPcmMessageBody
from event_bus.message_body.speaker_verificate_message_body import (
    SpeakerVerificateMessageBody,
)
from event_bus.processor.video_to_frame_processor import VideoToFrameProcessor
from event_bus.store.audio_to_pcm_store import AudioToPcmStore
from event_bus.store.video_to_frame_store import VideoToFrameStore
from store.local_store import LocalStore


class AudioToPcmProcessor(BaseEventBusProcessor):
    """音频转 PCM 帧处理器"""

    def __init__(self, processor_name: str):
        super().__init__(processor_name)
        self.audio_to_pcm_sample_rate: int = self.processor_properties[
            "audio_to_pcm_sample_rate"
        ]
        self.frame_length: int = self.processor_properties["frame_length"]
        self.frame_step: int = self.processor_properties["frame_step"]
        self.store = AudioToPcmStore(LocalStore.create)

    def _capture(
        self, audio_path: str
    ) -> Generator[tuple[torch.Tensor, int, int, int, int], None, None]:
        # 读取音频文件
        # audio: (num_channels, num_samples)
        audio: torch.Tensor
        sample_rate: int
        audio, sample_rate = torchaudio.load(audio_path)  # type: ignore
        if sample_rate != self.audio_to_pcm_sample_rate:
            # 重采样
            audio = torchaudio.transforms.Resample(
                sample_rate, self.audio_to_pcm_sample_rate
            )(audio)
        # 取单声道 (num_samples,)
        audio = audio[0]  # 采样值在 -1 到 1 之间

        # 分帧
        for i in range(0, len(audio) + self.frame_step, self.frame_step):
            if i - self.frame_length + 1 < 0:
                audio_frame = torch.cat(
                    [
                        torch.zeros(
                            self.frame_length - i - 1,
                            dtype=audio.dtype,
                            device=audio.device,
                        ),
                        audio[: i + 1],
                    ]
                )
            elif i + 1 > len(audio):
                audio_frame = torch.cat(
                    [
                        audio[i - self.frame_length + 1 :],
                        torch.zeros(
                            i + 1 - len(audio),
                            dtype=audio.dtype,
                            device=audio.device,
                        ),
                    ]
                )
            else:
                audio_frame = audio[i - self.frame_length + 1 : i + 1]
            yield (
                audio_frame,  # pcm
                self.audio_to_pcm_sample_rate,  # sample_rate
                self.frame_length,  # frame_length
                int(i / self.frame_step) + 1,  # frame_count
                int((i / self.audio_to_pcm_sample_rate) * 1000),  # frame_timestamp
            )

    def process(self, event_message_body: AudioToPcmMessageBody):
        # 发送消息
        for (
            pcm,
            sample_rate,
            frame_length,
            frame_count,
            frame_timestamp,
        ) in self._capture(event_message_body.audio_path):
            self.store.save_frame(
                self.get_request_id(),
                pcm,
                sample_rate,
                frame_length,
                self.frame_step,
                frame_count,
                frame_timestamp,
            )

            while (
                self._get_video_frame_timestamp_from_near_timestamp(
                    self.get_request_id(), frame_timestamp
                )
                is None
            ):
                # 等待对应视频帧有了之后，再继续，否则后面 asd 和 reduce 的时候会因为找不到对应的视频帧而卡住
                # TODO: 但如果视频帧一直没有，那么就会一直卡住，实时化的时候需要调整
                time.sleep(0.01)

            self.publish_next(
                "asd_topic",
                AsdMessageBody(
                    type="A",
                    audio_pcm=pcm,
                    audio_sample_rate=sample_rate,
                    audio_frame_length=frame_length,
                    audio_frame_step=self.frame_step,
                    audio_frame_count=frame_count,
                    audio_frame_timestamp=frame_timestamp,
                ),
            )
            self.publish_next(
                "speaker_verificate_topic",
                SpeakerVerificateMessageBody(
                    audio_pcm=pcm,
                    audio_sample_rate=sample_rate,
                    audio_frame_length=frame_length,
                    audio_frame_step=self.frame_step,
                    audio_frame_count=frame_count,
                    audio_frame_timestamp=frame_timestamp,
                ),
            )

    def process_exception(
        self, event_message_body: AudioToPcmMessageBody, exception: Exception
    ):
        raise Exception("AudioToPcmProcessor process_exception", exception)

    def _get_video_to_frame_store(self) -> VideoToFrameStore:
        """获取视频帧存储器"""
        video_to_frame_store = get_processor(
            config.event_bus["processors"]["VideoToFrameProcessor"]["processor_name"]
        )
        assert isinstance(video_to_frame_store, VideoToFrameProcessor)
        return video_to_frame_store.store

    def _get_video_frame_timestamp_from_near_timestamp(
        self, request_id: str, timestamp: int
    ) -> Optional[int]:
        """根据时间戳获取最近的视频帧时间戳"""
        video_to_frame_store = self._get_video_to_frame_store()
        frame_timestamp = video_to_frame_store.get_frame_timestamp_from_near_timestamp(
            request_id, timestamp
        )
        return frame_timestamp
