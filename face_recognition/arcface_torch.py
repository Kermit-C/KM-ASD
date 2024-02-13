#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸识别
@Author: chenkeming
@Date: 2024-02-13 19:27:44
"""

import timeit
from typing import Optional, Union

import cv2
import numpy as np
import torch
from skimage import transform as trans

from .arcface.backbones import get_model


class ArcFaceRecognizer:
    def __init__(
        self, trained_model: str, network: str = "r18", cpu: bool = False
    ) -> None:
        self.trained_model = trained_model
        self.network = network
        self.cpu = cpu
        torch.set_grad_enabled(False)
        self.net = get_model(network, fp16=False)
        self.net.load_state_dict(torch.load(trained_model))
        self.net.eval()
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = self.net.to(self.device)

    def _align_face(self, image, size, lmks):
        dst_w = size[1]
        dst_h = size[0]
        # landmark calculation of dst images
        base_w = 96
        base_h = 112
        assert dst_w >= base_w
        assert dst_h >= base_h
        base_lmk = [
            30.2946,
            51.6963,
            65.5318,
            51.5014,
            48.0252,
            71.7366,
            33.5493,
            92.3655,
            62.7299,
            92.2041,
        ]

        dst_lmk = np.array(base_lmk).reshape((5, 2)).astype(np.float32)
        if dst_w != base_w:
            slide = (dst_w - base_w) / 2
            dst_lmk[:, 0] += slide

        if dst_h != base_h:
            slide = (dst_h - base_h) / 2
            dst_lmk[:, 1] += slide

        src_lmk = lmks
        # using skimage method
        tform = trans.SimilarityTransform()
        tform.estimate(src_lmk, dst_lmk)
        t = tform.params[0:2, :]

        assert image.shape[2] == 3

        dst_image = cv2.warpAffine(image.copy(), t, (dst_w, dst_h))
        dst_pts = self._get_affine_points(src_lmk, t)
        return dst_image, dst_pts

    def _get_affine_points(self, pts_in, trans):
        pts_out = pts_in.copy()
        assert pts_in.shape[1] == 2

        for k in range(pts_in.shape[0]):
            pts_out[k, 0] = (
                pts_in[k, 0] * trans[0, 0] + pts_in[k, 1] * trans[0, 1] + trans[0, 2]
            )
            pts_out[k, 1] = (
                pts_in[k, 0] * trans[1, 0] + pts_in[k, 1] * trans[1, 1] + trans[1, 2]
            )
        return pts_out

    def gen_feat(
        self, img: Union[str, np.ndarray], face_lmks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """生成 l2 归一化后的特征
        :param img: 图片路径或者图片数据
        :param face_lmks: 人脸关键点，shape=(5, 2)，关键点定位+人脸对齐转为标准脸
        """
        if img is None:
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img)
            img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if face_lmks is not None:
            img, _ = self._align_face(img, (112, 112), face_lmks)
            img = np.ascontiguousarray(img)

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        img = img.to(self.device)

        start = timeit.default_timer()
        feat = self.net(img).detach().cpu().numpy()
        stop = timeit.default_timer()
        print("Time: %.2f s. " % (stop - start))
        feat /= np.sqrt(np.sum(feat**2, -1, keepdims=True))  # l2 norm

        return feat

    def calc_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算两个特征的相似度，0 为最不相似，1 为最相似"""
        return np.dot(feat1[0], feat2[0])
