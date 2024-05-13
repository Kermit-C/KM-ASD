#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸识别存储
@Author: Kermit
@Date: 2024-02-19 17:10:54
"""

from typing import Callable

import numpy as np

from store.store import Store


class FaceRecognitionStore:
    """人脸识别存储"""

    def __init__(
        self,
        store_creater: Callable[[bool, int], Store],
        max_face_count: int = 1000,
    ):
        self.store_creater = store_creater
        self.store_of_label = store_creater(True, max_face_count)

    def has_feat(self, label: str) -> bool:
        return self.store_of_label.has(label)

    def refresh_feat_lru(self, label: str):
        self.store_of_label.get(label)

    def save_feat(self, label: str, feat: np.ndarray):
        self.store_of_label.put(label, feat)

    def get_all_feats(self) -> tuple[list[str], list[np.ndarray]]:
        entries = self.store_of_label.get_all_entries()
        labels = [entry[0] for entry in entries]
        feats = [entry[1] for entry in entries]
        return labels, feats
