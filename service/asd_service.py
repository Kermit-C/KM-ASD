#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-22 10:44:41
"""

import random
from typing import Optional

import numpy as np

import config
from active_speaker_detection.asd_config import inference_params as infer_config
from active_speaker_detection.asd_inference import ActiveSpeakerDetector
from store.local_store import LocalStore

from .store.asd_store import ActiveSpeakerDetectionStore

detector: Optional[ActiveSpeakerDetector] = None
asd_store: ActiveSpeakerDetectionStore


def load_detector():
    global detector
    if detector is not None:
        return detector
    detector = ActiveSpeakerDetector(
        trained_model=config.asd_model,
        cpu=config.asd_cpu,
    )
    return detector


def load_asd_store():
    global asd_store
    asd_store = ActiveSpeakerDetectionStore(LocalStore.create, max_request_count=1000)
    return asd_store


def detect_active_speaker(
    request_id: str,
    frame_count: int,
    faces: list[np.ndarray],
    face_bboxes: list[tuple[int, int, int, int]],
    audio: np.ndarray,
) -> list[bool]:
    """
    :param request_id: request id，需要区分不同的请求，因为需要保存上下文，构造图
    :param frame_count: 帧计数，编号作用
    :param faces: 某刻画面中的人脸列表
    :param face_bboxes: 人脸列表对应的人脸框列表
    :param audio: audio, shape (n_samples,), float32 pcm
    """
    if detector is None:
        raise ValueError("Detector is not loaded")

    for face, face_bbox in zip(faces, face_bboxes):
        asd_store.save_frame(
            request_id=request_id,
            frame_count=frame_count,
            face_frame=face,
            face_bbox=face_bbox,
            audio_frame=audio,
        )

    faces_clip_list, audio_aggregate_list = get_faces_and_audios_of_graph(
        request_id=request_id, frame_count=frame_count
    )
    p_list: list[list[float]] = detector.detect_active_speaker(
        faces=faces_clip_list, audios=audio_aggregate_list
    )

    return [p > config.asd_p_threshold for p in p_list[-1]]


def get_faces_and_audios_of_graph(
    request_id: str,
    frame_count: int,
):
    frames_per_clip: int = infer_config["frmc"]
    ctx_size: int = infer_config["ctx"]
    n_clips: int = infer_config["nclp"]
    strd: int = infer_config["strd"]

    # (frames_per_clip + (n_clips - 1) * strd, ctx_size)
    faces_list: list[list[Optional[np.ndarray]]] = [
        [None for _ in range(ctx_size)]
        for _ in range(frames_per_clip + (n_clips - 1) * strd)
    ]
    audio_list: list[Optional[np.ndarray]] = [
        None for i_ in range(frames_per_clip + (n_clips - 1) * strd)
    ]
    n_clip_mask = [frames_per_clip - 1 + i * strd for i in range(n_clips)]

    # 取 frame
    # 取人脸
    last_face_boxes = None
    for i in range(frames_per_clip + (n_clips - 1) * strd - 1, -1, -1):
        curr_frame_count = (
            frame_count - (frames_per_clip + (n_clips - 1) * strd) + i + 1
        )
        frame_faces = asd_store.get_frame_faces(request_id, curr_frame_count)
        if len(frame_faces) == 0:
            # 如果没有人脸，就用下一帧一样的
            faces_list[i] = faces_list[i + 1]
            continue

        frame_faces_idx_list = list(range(ctx_size))
        if last_face_boxes is None:
            # 最新帧，直接分配就好
            for j in range(ctx_size):
                if j < len(frame_faces):
                    frame_faces_idx_list[j] = j
                else:
                    frame_faces_idx_list[j] = random.randint(0, len(frame_faces) - 1)
        else:
            # 先根据 IOU 找到最接近的人脸，分配
            alloc_num = 0
            for j, frame_face in enumerate(frame_faces):
                face_bbox = frame_face["face_bbox"]
                face_idx = get_face_idx_from_last_frame(last_face_boxes, face_bbox)
                if face_idx != -1:
                    frame_faces_idx_list[j] = face_idx
                    alloc_num += 1
            # 没有分配到的，随机分配
            if alloc_num > 0:
                for j in range(ctx_size - alloc_num, ctx_size):
                    frame_faces_idx_list[j] = frame_faces_idx_list[
                        random.randint(0, alloc_num - 1)
                    ]
            else:
                # 如果没有找到最接近的人脸，就代表切画面了，直接用下一帧一样的
                faces_list[i] = faces_list[i + 1]
                continue

        faces_list[i] = [
            frame_faces[frame_faces_idx_list[j]]["face_frame"] for j in range(ctx_size)
        ]
        last_face_boxes = [
            frame_faces[frame_faces_idx_list[j]]["face_bbox"] for j in range(ctx_size)
        ]
    # 取音频
    for i in range(frames_per_clip + (n_clips - 1) * strd - 1, -1, -1):
        curr_frame_count = (
            frame_count - (frames_per_clip + (n_clips - 1) * strd) + i + 1
        )
        while audio_list[i] is None:
            audio_list[i] = asd_store.get_frame_audio(request_id, curr_frame_count)
            curr_frame_count += 1

    # 合 frame
    # (n_clips, ctx_size, frames_per_clip)
    faces_clip_list: list[list[list[np.ndarray]]] = []
    # (n_clips,)
    audio_aggregate_list: list[np.ndarray] = []
    for i in range(
        frames_per_clip + (n_clips - 1) * strd - 1, frames_per_clip - 1 - 1, -1
    ):
        if i not in n_clip_mask:
            continue

        faces_clip = []
        for j in range(ctx_size):
            faces_clip = [i[j] for i in faces_list[i - frames_per_clip + 1 : i + 1]]
        faces_clip_list.insert(0, faces_clip)  # type: ignore

        aggregate_audio_list = []
        for j in range(frames_per_clip):
            if (
                len(aggregate_audio_list) > 0
                and aggregate_audio_list[0] == audio_list[i - j]
            ):
                # 如果当前帧和上一帧的音频一样，就不用再取了
                break
            aggregate_audio_list.insert(0, audio_list[i - j])
        audio_aggregate_list.insert(0, np.concatenate(aggregate_audio_list, axis=0))

    return faces_clip_list, audio_aggregate_list


def get_face_idx_from_last_frame(
    face_bboxes: list[tuple[int, int, int, int]], face_bbox: tuple[int, int, int, int]
) -> int:
    """从上一帧人脸中找到最接近的人脸"""
    for i, last_frame_face_bbox in enumerate(face_bboxes):
        # 计算两个人脸框的IOU
        iou = get_iou(face_bbox, last_frame_face_bbox)
        if iou > config.asd_same_face_between_frames_iou_threshold:
            return i
    return -1


def get_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """计算两个人脸框的IOU"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    intersection = x_overlap * y_overlap
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection
    return intersection / union
