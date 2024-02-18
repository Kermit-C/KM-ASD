#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 事件总线初始化
@Author: chenkeming
@Date: 2024-02-10 15:50:27
"""

from config import event_bus
from event_bus.event_bus_factory import create_processor
from event_bus.processor.asd_processor import AsdProcessor
from event_bus.processor.audio_to_pcm_processor import AudioToPcmProcessor
from event_bus.processor.face_crop_processor import FaceCropProcessor
from event_bus.processor.face_detect_processor import FaceDetectProcessor
from event_bus.processor.face_recognize_processor import FaceRecognizeProcessor
from event_bus.processor.output_processor import OutputProcessor
from event_bus.processor.speaker_verificate_processor import SpeakerVerificateProcessor
from event_bus.processor.video_to_frame_processor import VideoToFrameProcessor


def init_event_bus():
    """初始化事件总线"""
    create_processor(
        VideoToFrameProcessor(
            event_bus["processors"][VideoToFrameProcessor]["processor_name"]
        ),
        event_bus["publisher"]["name"],
        event_bus["processors"][VideoToFrameProcessor]["topic"],
    )
    create_processor(
        AudioToPcmProcessor(
            event_bus["processors"][AudioToPcmProcessor]["processor_name"]
        ),
        event_bus["publisher"]["name"],
        event_bus["processors"][AudioToPcmProcessor]["topic"],
    )
    create_processor(
        FaceDetectProcessor(
            event_bus["processors"][FaceDetectProcessor]["processor_name"]
        ),
        event_bus["publisher"]["name"],
        event_bus["processors"][FaceDetectProcessor]["topic"],
    )
    create_processor(
        FaceCropProcessor(event_bus["processors"][FaceCropProcessor]["processor_name"]),
        event_bus["publisher"]["name"],
        event_bus["processors"][FaceCropProcessor]["topic"],
    )
    create_processor(
        FaceRecognizeProcessor(
            event_bus["processors"][FaceRecognizeProcessor]["processor_name"]
        ),
        event_bus["publisher"]["name"],
        event_bus["processors"][FaceRecognizeProcessor]["topic"],
    )
    create_processor(
        AsdProcessor(event_bus["processors"][AsdProcessor]["processor_name"]),
        event_bus["publisher"]["name"],
        event_bus["processors"][AsdProcessor]["topic"],
    )
    create_processor(
        SpeakerVerificateProcessor(
            event_bus["processors"][SpeakerVerificateProcessor]["processor_name"]
        ),
        event_bus["publisher"]["name"],
        event_bus["processors"][SpeakerVerificateProcessor]["topic"],
    )
    create_processor(
        OutputProcessor(event_bus["processors"][OutputProcessor]["processor_name"]),
        event_bus["publisher"]["name"],
        event_bus["processors"][OutputProcessor]["topic"],
    )
