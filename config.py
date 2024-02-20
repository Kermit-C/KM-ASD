#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 全局配置
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from speaker_verification.ecapa_tdnn_config import Args as SpeakerVerificationArgs

model_service_server_grpc_port = 50051
model_service_server_max_workers = 200

extract_audio_track_sample_rate = 16000
render_video_path = "tmp/render"

face_detection_model = "face_detection/retinaface_weights/mobilenet0.25_Final.pth"
face_detection_network = "mobile0.25"
face_detection_cpu = False

face_recognize_model = "face_recognition/arcface_weights/ms1mv3_r18_backbone.pth"
face_recognize_network = "r18"
face_recognize_cpu = False
face_recognize_sim_threshold = 0.5

speaker_verificate_cpu = False
speaker_verificate_score_threshold = SpeakerVerificationArgs.threshold


# 事件总线配置
event_bus = {
    "publisher": {
        "name": "default_event_bus_publisher",
    },
    "processors": {
        "VideoToFrameProcessor": {
            "processor_name": "video_to_frame_processor",
            "topic": "video_to_frame_topic",
            "timeout": 60,  # 暂未使用
            "properties": {
                "target_video_fps": 30,
            },
        },
        "AudioToPcmProcessor": {
            "processor_name": "audio_to_pcm_processor",
            "topic": "audio_to_pcm_topic",
            "timeout": 60,  # 暂未使用
            "properties": {
                "audio_to_pcm_sample_rate": 16000,
                "frame_length": 8000,  # 500ms
                "frame_step": 480,  # 30ms
            },
        },
        "FaceDetectProcessor": {
            "processor_name": "face_detect_processor",
            "topic": "face_detect_topic",
            "timeout": 2 * (1 / 30),  # 2帧时间
            "properties": {
                # "face_detect_model_path": "models/face_detect_model",
            },
        },
        "FaceCropProcessor": {
            "processor_name": "face_crop_processor",
            "topic": "face_crop_topic",
            "timeout": 1,
            "properties": {
                "face_crop_size": 112,
            },
        },
        "FaceRecognizeProcessor": {
            "processor_name": "face_recognize_processor",
            "topic": "face_recognize_topic",
            "timeout": 5 * (1 / 30),  # 5帧时间
            "properties": {
                # "face_recognize_model_path": "models/face_recognize_model",
            },
        },
        "AsdProcessor": {
            "processor_name": "asd_processor",
            "topic": "asd_topic",
            "timeout": 5 * (1 / 30),  # 5帧时间
            "properties": {
                # "asd_model_path": "models/asd_model",
            },
        },
        "SpeakerVerificateProcessor": {
            "processor_name": "speaker_verificate_processor",
            "topic": "speaker_verificate_topic",
            "timeout": 5 * (1 / 30),  # 5帧时间
            "properties": {
                # "speaker_verificate_model_path": "models/speaker_verificate_model",
            },
        },
        "ReduceProcessor": {
            "processor_name": "reduce_processor",
            "topic": "reduce_topic",
            "timeout": 1 / 30,  # 1帧时间 暂未使用
            "properties": {
                # "output_path": "output",
            },
        },
    },
}
