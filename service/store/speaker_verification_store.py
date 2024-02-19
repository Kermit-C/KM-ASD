#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 说话人验证存储
@Author: Kermit
@Date: 2024-02-19 17:37:30
"""

from typing import Callable

import torch

from store.store import Store


class SpeakerVerificationStore:
    """说话人验证存储"""

    def __init__(
        self,
        store_creater: Callable[[bool, int], Store],
        max_face_count: int = 1000,
    ):
        self.store_creater = store_creater
        self.store_of_label = store_creater(True, max_face_count)

    def has_feat(self, label: str) -> bool:
        return self.store_of_label.has(label)

    def save_feat(self, label: str, feat: torch.Tensor):
        self.store_of_label.put(label, feat)

    def get_all_feats(self) -> tuple[list[str], list[torch.Tensor]]:
        entries = self.store_of_label.get_all_entries()
        labels = [entry[0] for entry in entries]
        feats = [entry[1] for entry in entries]
        return labels, feats
