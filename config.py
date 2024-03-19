#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 全局配置
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from active_speaker_detection.asd_config import inference_params as AsdInferenceParams
from speaker_verification.ecapa_tdnn_config import Args as SpeakerVerificationArgs

########## 模型服务配置 ##########

model_service_server_grpc_port = 50051
model_service_server_max_workers = 100
model_service_server_asd_max_workers = 1
model_service_server_asd_worker_wait_timeout = 10
model_service_server_face_detection_max_workers = 1
model_service_server_face_detection_worker_wait_timeout = 10
model_service_server_face_recognize_max_workers = 1
model_service_server_face_recognize_worker_wait_timeout = 10
model_service_server_speaker_verificate_max_workers = 1
model_service_server_speaker_verificate_worker_wait_timeout = 10
model_service_server_thread_name_prefix = "model_service_server"

extract_audio_track_sample_rate = 16000
render_video_path = "tmp/render"

asd_enabled = True
asd_model = "active_speaker_detection/results/22.pth"
asd_cpu = False
asd_p_threshold = 0.5
asd_same_face_between_frames_iou_threshold = 0.5

face_detection_enabled = True
face_detection_model = "face_detection/retinaface_weights/mobilenet0.25_Final.pth"
face_detection_network = "mobile0.25"
face_detection_cpu = False
face_detection_confidence_threshold = 0.5

face_recognize_enabled = True
face_recognize_model = "face_recognition/arcface_weights/ms1mv3_r18_backbone.pth"
face_recognize_network = "r18"
face_recognize_cpu = False
face_recognize_sim_threshold = 0.5
face_recognize_register_path = "tmp/register/faces"

speaker_verificate_enabled = True
speaker_verificate_cpu = False
speaker_verificate_sample_rate = 16000
speaker_verificate_score_threshold = SpeakerVerificationArgs.threshold
speaker_verificate_register_path = "tmp/register/speakers"


########## 接入服务配置 ##########

# 事件总线配置
event_bus_executor_max_workers = 200
event_bus_executor_thread_name_prefix = "event_bus_executor"
event_bus = {
    "publisher": {
        "name": "default_event_bus_publisher",
    },
    "processors": {
        "VideoToFrameProcessor": {
            "processor_name": "video_to_frame_processor",
            "topic": "video_to_frame_topic",
            "timeout": 6000,  # 暂未使用
            "properties": {
                "target_video_fps": 30,
            },
        },
        "AudioToPcmProcessor": {
            "processor_name": "audio_to_pcm_processor",
            "topic": "audio_to_pcm_topic",
            "timeout": 6000,  # 暂未使用
            "properties": {
                "audio_to_pcm_sample_rate": 16000,
                "frame_length": 8000,  # 500ms
                "frame_step": 480,  # 30ms
            },
        },
        "FaceDetectProcessor": {
            "processor_name": "face_detect_processor",
            "topic": "face_detect_topic",
            "timeout": 1,  # 3 * (1 / 30),  # 3帧时间
            "properties": {
                "detect_lag": 3,  # 3 帧
                "same_face_between_frames_iou_threshold": 0.5,
            },
        },
        "FaceCropProcessor": {
            "processor_name": "face_crop_processor",
            "topic": "face_crop_topic",
            "timeout": 3 * (1 / 30),  # 3帧时间
            "properties": {
                "face_crop_size": 112,
                "same_face_between_frames_iou_threshold": 0.5,
            },
        },
        "FaceRecognizeProcessor": {
            "processor_name": "face_recognize_processor",
            "topic": "face_recognize_topic",
            "timeout": 1,  # 5 * (1 / 30),  # 5帧时间
            "properties": {
                "same_face_between_frames_iou_threshold": 0.5,
            },
        },
        "AsdProcessor": {
            "processor_name": "asd_processor",
            "topic": "asd_topic",
            "timeout": 1,  # 10 * (1 / 30),  # 10帧时间
            "properties": {
                "frmc": AsdInferenceParams["frmc"],
                "detect_lag": AsdInferenceParams["strd"],
            },
        },
        "SpeakerVerificateProcessor": {
            "processor_name": "speaker_verificate_processor",
            "topic": "speaker_verificate_topic",
            "timeout": 1,  # 5 * (1 / 30),  # 5帧时间
            "properties": {
                "aggregate_frame_length": 24000,  # 1500ms
            },
        },
        "ReduceProcessor": {
            "processor_name": "reduce_processor",
            "topic": "reduce_topic",
            "timeout": 6000,  # 暂未使用
            "properties": {
                # "output_path": "output",
            },
        },
    },
}
