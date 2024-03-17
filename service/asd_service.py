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
    frame_height: int,
    frame_width: int,
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

    # 人脸为空直接返回
    if len(faces) == 0:
        return []

    for face, face_bbox in zip(faces, face_bboxes):
        asd_store.save_frame(
            request_id=request_id,
            frame_count=frame_count,
            face_frame=face,
            face_bbox=face_bbox,
            audio_frame=audio,
        )

    (
        faces_clip_list,
        audio_aggregate_list,
        faces_rbbox_clip_list,
        faces_bbox_clip_list,
    ) = get_faces_and_audios_of_graph(
        request_id=request_id,
        frame_count=frame_count,
        frame_height=frame_height,
        frame_width=frame_width,
    )

    # TODO: 返回 emb，缓存 emb
    p_list: list[list[float]] = detector.detect_active_speaker(
        faces=faces_clip_list,
        audios=audio_aggregate_list,
        face_bboxes=faces_rbbox_clip_list,
    )

    request_faces_result_idxes = []
    for face_bbox in face_bboxes:
        try:
            idx = faces_bbox_clip_list[-1].index(face_bbox)
            request_faces_result_idxes.append(idx)
        except ValueError:
            request_faces_result_idxes.append(-1)

    return [
        p_list[-1][i] > config.asd_p_threshold if i != -1 else False
        for i in request_faces_result_idxes
    ]


def get_faces_and_audios_of_graph(
    request_id: str,
    frame_count: int,
    frame_height: int,
    frame_width: int,
) -> tuple[
    list[list[list[np.ndarray]]],
    list[np.ndarray],
    list[list[tuple[float, float, float, float]]],
    list[list[tuple[int, int, int, int]]],
]:
    frames_per_clip: int = infer_config["frmc"]
    max_ctx_size: int = (
        infer_config["ctx"] * 3
    )  # 三倍的上下文大小，一次请求可以分治成三份分别请求算法
    n_clips: int = infer_config["nclp"]
    strd: int = infer_config["strd"]

    # TODO: 这里刚好覆盖了 n_clips 时间步长，但图的消息传递每层要再往外 n_clips * strd 个时间步长，所以总共是需要 n * n_clips * strd 个时间步长的（n 是图的层数）
    # (frames_per_clip + (n_clips - 1) * strd, max_ctx_size), 矩阵窗口
    faces_list: list[list[Optional[np.ndarray]]] = [
        [None for _ in range(max_ctx_size)]
        for _ in range(frames_per_clip + (n_clips - 1) * strd)
    ]
    faces_bbox_list: list[list[Optional[tuple[int, int, int, int]]]] = [
        [None for _ in range(max_ctx_size)]
        for _ in range(frames_per_clip + (n_clips - 1) * strd)
    ]
    audio_list: list[Optional[np.ndarray]] = [
        None for i_ in range(frames_per_clip + (n_clips - 1) * strd)
    ]
    n_clip_mask = [frames_per_clip - 1 + i * strd for i in range(n_clips)]

    # 取 frame
    # 取人脸
    last_face_boxes: Optional[list[Optional[tuple[int, int, int, int]]]] = None
    for i in range(frames_per_clip + (n_clips - 1) * strd - 1, -1, -1):
        curr_frame_count = (
            frame_count - (frames_per_clip + (n_clips - 1) * strd) + i + 1
        )
        frame_faces = asd_store.get_frame_faces(request_id, curr_frame_count)
        if len(frame_faces) == 0:
            # 如果没有人脸，就用下一帧一样的
            faces_list[i] = faces_list[i + 1]
            faces_bbox_list[i] = faces_bbox_list[i + 1]
            continue

        frame_faces_idx_list = list(range(max_ctx_size))
        if last_face_boxes is None:
            # 最新帧，直接分配就好
            for j in range(max_ctx_size):
                if j < len(frame_faces):
                    frame_faces_idx_list[j] = j
                else:
                    # frame_faces_idx_list[j] = random.randint(0, len(frame_faces) - 1)
                    frame_faces_idx_list[j] = -1
        else:
            # 先根据 IOU 找到最接近的人脸，分配
            alloc_num = 0
            for j, frame_face in enumerate(frame_faces):
                face_bbox = frame_face["face_bbox"]
                last_face_idx = get_face_idx_from_last_frame(last_face_boxes, face_bbox)  # type: ignore
                if last_face_idx != -1:
                    frame_faces_idx_list[last_face_idx] = j
                    alloc_num += 1
            if alloc_num > 0:
                # # 没有分配到的，随机分配
                # for j in range(alloc_num, max_ctx_size):
                #     frame_faces_idx_list[j] = frame_faces_idx_list[
                #         random.randint(0, alloc_num - 1)
                #     ]
                # 没有分配到的，不分配
                # TODO: 这里遗漏了在最新帧未出现、但在上一帧出现的人脸，按理说应当保留，但这里的矩阵窗口第二维度是固定身份的，所以这里暂不处理了
                for j in range(alloc_num, max_ctx_size):
                    frame_faces_idx_list[j] = -1
            else:
                # 如果没有找到最接近的人脸，就代表切画面了，直接用下一帧一样的
                faces_list[i] = faces_list[i + 1]
                faces_bbox_list[i] = faces_bbox_list[i + 1]
                continue

        faces_list[i] = [
            (
                frame_faces[frame_faces_idx_list[j]]["face_frame"]
                if frame_faces_idx_list[j] != -1
                else None
            )
            for j in range(max_ctx_size)
        ]
        faces_bbox_list[i] = [
            (
                frame_faces[frame_faces_idx_list[j]]["face_bbox"]
                if frame_faces_idx_list[j] != -1
                else None
            )
            for j in range(max_ctx_size)
        ]
        last_face_boxes = faces_bbox_list[i]
    # 取音频
    for i in range(frames_per_clip + (n_clips - 1) * strd - 1, -1, -1):
        curr_frame_count = (
            frame_count - (frames_per_clip + (n_clips - 1) * strd) + i + 1
        )
        while audio_list[i] is None:
            audio_list[i] = asd_store.get_frame_audio(request_id, curr_frame_count)
            curr_frame_count += 1

    # 合 frame
    # (n_clips, max_ctx_size, frames_per_clip)
    faces_clip_list: list[list[list[np.ndarray]]] = []
    faces_bbox_clip_list: list[list[tuple[int, int, int, int]]] = []
    # (n_clips,)
    audio_aggregate_list: list[np.ndarray] = []
    is_last_frame = True
    for i in range(
        frames_per_clip + (n_clips - 1) * strd - 1, frames_per_clip - 1 - 1, -1
    ):
        if i not in n_clip_mask:
            continue

        faces_clip: list[list[np.ndarray]] = []
        faces_bbox: list[tuple[int, int, int, int]] = []
        for j in range(max_ctx_size):

            # 如果是最新帧，而且没有不为 None 的 clip，就复制一份上一实体的
            if is_last_frame and any(
                [faces_list[k][j] is None for k in range(len(faces_list))]
            ):
                for k in range(len(faces_list)):
                    faces_list[k][j] = faces_list[k][j - 1]
                    faces_bbox_list[k][j] = faces_bbox_list[k][j - 1]

            if any(
                [
                    faces_list[k][j] is None
                    for k in range(i - frames_per_clip + 1, i + 1)
                ]
            ):
                # clips 全是 None，复制上一帧的
                faces_clip.append(faces_clip_list[0][j])
                faces_bbox.append(faces_bbox_clip_list[0][j])
            else:
                # 取不为 None 的 clips
                curr_faces_list = list(
                    filter(
                        lambda x: x[j] is not None,
                        faces_list[i - frames_per_clip + 1 : i + 1],
                    )
                )
                # 复制首部的人脸，直到填满
                while len(curr_faces_list) < frames_per_clip:
                    curr_faces_list.insert(0, curr_faces_list[0])

                faces_clip.append([k[j] for k in curr_faces_list])  # type: ignore
                faces_bbox.append(faces_bbox_list[i][j])  # type: ignore
        faces_clip_list.insert(0, faces_clip)
        faces_bbox_clip_list.insert(0, faces_bbox)

        aggregate_audio_list = []
        for j in range(frames_per_clip):
            if (
                len(aggregate_audio_list) > 0
                and aggregate_audio_list[0] is audio_list[i - j]
            ):
                # 如果当前帧和上一帧的音频一样，就不用再取了
                break
            aggregate_audio_list.insert(0, audio_list[i - j])
        audio_aggregate_list.insert(0, np.concatenate(aggregate_audio_list, axis=0))

        is_last_frame = False

    # faces_bbox_clip_list 转换成相对坐标
    faces_rbbox_clip_list = []
    for faces_bbox_clip in faces_bbox_clip_list:
        rbbox_list = []
        for face_bbox in faces_bbox_clip:
            x1, y1, x2, y2 = face_bbox
            rbbox_list.append(
                (
                    x1 / frame_width,
                    y1 / frame_height,
                    x2 / frame_width,
                    y2 / frame_height,
                )
            )
        faces_rbbox_clip_list.append(rbbox_list)

    return (
        faces_clip_list,
        audio_aggregate_list,
        faces_rbbox_clip_list,
        faces_bbox_clip_list,
    )


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
