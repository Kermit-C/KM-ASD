#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-19 17:49:45
"""

from threading import RLock
from typing import Callable, Optional

import numpy as np
import torch

from store.store import Store


class ReduceStore:
    def __init__(
        self,
        store_creater: Callable[[bool, int], Store],
        max_request_count: int = 1000,
    ):
        self.store_creater = store_creater
        self.store_of_request = store_creater(True, max_request_count)
        self.save_frame_lock = RLock()
        self.frame_resulted_count_lock = RLock()

    def save_frame_result(
        self,
        request_id: str,
        frame_count: int,
        frame_timestamp: int,
        # ASD & FACE_RECOGNIZE COMMON
        frame_face_idx: Optional[int] = None,  # 人脸索引，一个视频帧中可能有多个人脸
        # ASD
        frame_face_count: Optional[int] = None,  # 人脸数量
        frame_face_bbox: Optional[tuple[int, int, int, int]] = None,
        frame_face_asd_status: Optional[int] = None,
        # FACE_RECOGNIZE
        frame_face_label: Optional[str] = None,
        # SPEAKER_VERIFICATE
        frame_voice_label: Optional[str] = None,
    ):
        """保存帧结果"""
        with self.save_frame_lock:
            if not self.store_of_request.has(request_id):
                self.store_of_request.put(
                    request_id, {"frame_results": [], "frame_resulted_count": 0}
                )
            request_store = self.store_of_request.get(request_id)

            while len(request_store["frame_results"]) <= frame_count - 1:
                # 补充空帧
                request_store["frame_results"].append(None)

            if request_store["frame_results"][frame_count - 1] is None:
                request_store["frame_results"][frame_count - 1] = {
                    "frame_timestamp": (
                        frame_timestamp if frame_timestamp is not None else -1
                    ),
                    "frame_face_count": (
                        frame_face_count if frame_face_count is not None else -1
                    ),
                    "frame_face_idx": [],
                    "frame_face_bbox": [],
                    "frame_face_asd_status": [],
                    "frame_face_label": [],
                    "frame_voice_label": [],
                    "is_complete": False,
                    "is_resulted": False,
                }
            frame_results = request_store["frame_results"][frame_count - 1]

            # ASD & FACE_RECOGNIZE COMMON
            idx = len(frame_results["frame_face_idx"])
            if frame_face_idx is not None:
                if frame_face_idx not in frame_results["frame_face_idx"]:
                    # 记录人脸索引，给多个结果提前 append 空值
                    frame_results["frame_face_idx"].append(frame_face_idx)
                    frame_results["frame_face_bbox"].append(None)
                    frame_results["frame_face_asd_status"].append(None)
                    frame_results["frame_face_label"].append(None)
                    idx = len(frame_results["frame_face_idx"]) - 1
                else:
                    # 人脸索引已经存在，直接取出索引
                    idx = frame_results["frame_face_idx"].index(frame_face_idx)

            if frame_face_count is not None:
                frame_results["frame_face_count"] = frame_face_count
            if frame_face_bbox is not None:
                frame_results["frame_face_bbox"][idx] = frame_face_bbox
            if frame_face_asd_status is not None:
                frame_results["frame_face_asd_status"][idx] = frame_face_asd_status
            if frame_face_label is not None:
                frame_results["frame_face_label"][idx] = frame_face_label

            # SPEAKER_VERIFICATE
            if frame_voice_label is not None:
                # 说话人验证结果只有一个
                frame_results["frame_voice_label"].append(frame_voice_label)

            # 判断帧结果是否完整
            frame_results["is_complete"] = self.is_frame_result_complete(
                request_id, frame_count
            )

    def is_frame_result_complete(self, request_id: str, frame_count: int) -> bool:
        """判断帧结果是否完整（只要人脸结果完成了就行，不管 SPEAKER_VERIFICATE）"""
        if not self.store_of_request.has(request_id):
            return False
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frame_results"]) < frame_count:
            return False
        frame_results = request_store["frame_results"][frame_count - 1]
        # 人脸索引数量等于人脸数量，且 ASD 结果、人脸识别都齐了，就是完整的
        is_complete = (
            frame_results is not None
            and len(frame_results["frame_face_idx"])
            == frame_results["frame_face_count"]
            and all(
                map(lambda x: x is not None, frame_results["frame_face_asd_status"])
            )
            and all(map(lambda x: x is not None, frame_results["frame_face_label"]))
        )
        return is_complete

    def is_frame_resulted(self, request_id: str, frame_count: int) -> bool:
        """判断帧结果是否处理完"""
        if not self.store_of_request.has(request_id):
            return False
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frame_results"]) < frame_count:
            return False
        return (
            request_store["frame_results"][frame_count - 1] is not None
            and request_store["frame_results"][frame_count - 1]["is_resulted"]
        )

    def is_frame_before_all_resulted(self, request_id: str, frame_count: int) -> bool:
        """判断帧之前的结果是否全部处理完"""
        if not self.store_of_request.has(request_id):
            return False
        request_store = self.store_of_request.get(request_id)
        return request_store["frame_resulted_count"] >= frame_count - 1

    def get_frame_after_all_complete_but_not_resulted(
        self, request_id: str, frame_count: int
    ) -> list[int]:
        """获取帧之后所有完整但未处理的帧"""
        if not self.store_of_request.has(request_id):
            return []
        request_store = self.store_of_request.get(request_id)
        result = []
        for i in range(frame_count, len(request_store["frame_results"])):
            frame_result = request_store["frame_results"][i]
            if frame_result is None:
                return result
            if not frame_result["is_complete"]:
                return result
            result.append(i + 1)
        return result

    def set_frame_resulted(self, request_id: str, frame_count: int):
        """设置帧结果已经处理"""
        if not self.store_of_request.has(request_id):
            return
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frame_results"]) < frame_count:
            return
        request_store["frame_results"][frame_count - 1]["is_resulted"] = True
        with self.frame_resulted_count_lock:
            request_store["frame_resulted_count"] += 1

    def get_frame_result(self, request_id: str, frame_count: int) -> dict:
        """获取帧结果"""
        empty_result = {
            "frame_timestamp": -1,
            "frame_face_count": -1,
            "frame_face_idx": [],
            "frame_face_bbox": [],
            "frame_face_asd_status": [],
            "frame_face_label": [],
            "frame_voice_label": [],
        }
        if not self.store_of_request.has(request_id):
            return empty_result
        request_store = self.store_of_request.get(request_id)
        if len(request_store["frame_results"]) < frame_count:
            return empty_result
        return request_store["frame_results"][frame_count - 1]
