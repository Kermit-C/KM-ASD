#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 人脸检测
@Author: chenkeming
@Date: 2024-02-09 17:24:59
"""

from __future__ import print_function

import os
from typing import Union

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .retinaface.layers.functions.prior_box import PriorBox
from .retinaface.models.retinaface import RetinaFace
from .retinaface.utils.box_utils import decode, decode_landm
from .retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from .retinaface.utils.timer import Timer
from .retinaface_config import cfg_mnet, cfg_re50


class RetinaFaceDetector:
    def __init__(
        self,
        trained_model: str,  # path to the trained model
        network: str,  # mobile0.25 or resnet50
        cpu: bool = False,
        confidence_threshold=0.02,  # 置信度阈值
        top_k=5000,
        nms_threshold=0.4,
        keep_top_k=750,
        vis_thres=0.5,  # visualization_threshold
    ) -> None:
        self.trained_model = trained_model
        self.network = network
        self.cpu = cpu
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_thres = vis_thres

        torch.set_grad_enabled(False)
        self.cfg = None
        if network == "mobile0.25":
            self.cfg = cfg_mnet
        elif network == "resnet50":
            self.cfg = cfg_re50

        # net and model
        self.net = RetinaFace(cfg=self.cfg, phase="test")
        self.net = self._load_model(self.net, trained_model, cpu)
        self.net.eval()
        print("Finished loading model!")
        # print(self.net)
        cudnn.benchmark = True
        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = self.net.to(self.device)

    def detect_faces(
        self,
        image_or_image_path: Union[cv2.typing.MatLike, np.ndarray, str],
        save_path: str = None,
    ):
        """
        :param image_or_image_path: 输入图片或图片路径
        :param save_path: 保存路径
        :return: 返回检测到的人脸信息，一个 list，每个元素是一个 tuple，包含以下信息：
            (xmin, ymin, xmax, ymax, score, w, h, eye1_x, eye1_y, eye2_x, eye2_y, nose_x, nose_y, mouth1_x, mouth1_y, mouth2_x, mouth2_y)
        """
        # 图片读取
        if isinstance(image_or_image_path, str):
            img_raw = cv2.imread(image_or_image_path, cv2.IMREAD_COLOR)
        elif isinstance(image_or_image_path, np.ndarray):
            img_raw = cv2.UMat(image_or_image_path)
        else:
            img_raw = image_or_image_path
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        scale = scale.to(self.device)

        _t = {"forward_pass": Timer(), "misc": Timer()}
        _t["forward_pass"].tic()
        # forward pass
        loc, conf, landms = self.net(img)
        _t["forward_pass"].toc()
        _t["misc"].tic()

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1][:top_k]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:keep_top_k, :]
        # landms = landms[:keep_top_k, :]

        # 把 landms 拼接到 dets 中
        dets = np.concatenate((dets, landms), axis=1)

        _t["misc"].toc()
        print(
            "forward_pass_time: {:.4f}s misc: {:.4f}s".format(
                _t["forward_pass"].average_time, _t["misc"].average_time
            )
        )

        # show image
        self._save_image(save_path, img_raw, dets)

        # xmin, ymin, xmax, ymax, score, w, h, eye1_x, eye1_y, eye2_x, eye2_y, nose_x, nose_y, mouth1_x, mouth1_y, mouth2_x, mouth2_y
        results = [
            (b[0], b[1], b[2], b[3], b[4], b[2] - b[0] + 1, b[3] - b[1] + 1, *b[5:15])
            for b in dets
        ]
        return results

    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print("Missing keys:{}".format(len(missing_keys)))
        print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        print("Used keys:{}".format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
        return True

    def _remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
        print("remove prefix '{}'".format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def _load_model(self, model, pretrained_path, load_to_cpu):
        print("Loading pretrained model from {}".format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
            )
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(
                pretrained_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, "module.")
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def _save_image(self, save_path, img_raw, dets):
        if save_path is not None:
            for b in dets:
                if b[4] < self.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))

                # Draw bounding box
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

                cx = b[0]
                cy = b[1] + 12
                cv2.putText(
                    img_raw,
                    text,
                    (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                )

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # save image
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            name = save_path
            cv2.imwrite(name, img_raw)
