#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 声纹识别，说话人验证
@Author: chenkeming
@Date: 2024-02-14 10:36:53
"""

import timeit
from typing import Union

import numpy as np
import torch

from .ecapa_tdnn.config import get_config
from .ecapa_tdnn.dataloader import loadWAV
from .ecapa_tdnn.loss import built_loss
from .ecapa_tdnn.model import built_model
from .ecapa_tdnn.SpeakerNet import *
from .ecapa_tdnn_config import Args


class EcapaTdnnVerificator:
    def __init__(self, cpu: bool = False) -> None:
        self.device = torch.device("cpu" if cpu else "cuda")
        cfg, _ = get_config(Args)
        torch.set_grad_enabled(False)
        self.model = built_model(cfg).to(self.device)
        loss = built_loss(cfg).to(self.device)
        self.model = SpeakerNet(cfg, model=self.model, loss=loss)
        self.model.load_state_dict(
            torch.load(cfg.MODEL.RESUME)["model_state_dict"], strict=False
        )
        self.model.eval()

        self.max_frames = 300
        self.split_num = 5

    def _parse_audio(
        self, audio: np.ndarray, max_frames: int, num_eval=10
    ) -> np.ndarray:
        max_audio = (max_frames - 2) * 160 + 240  # 160 是帧长，240 是帧移
        audiosize = audio.shape[0]
        if audiosize <= max_audio:
            # 给不足 max_frams 的 padding
            shortage = max_audio - audiosize + 1
            # 0 是前面的 padding，后面 padding shortage 本身的循环
            audio = np.pad(audio, (0, shortage), "wrap")
            audiosize = audio.shape[0]

        split_audio = []
        # 分为 10 段，分别计算 feat
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
        for asf in startframe:
            split_audio.append(audio[int(asf) : int(asf) + max_audio])
        parsed_audio = np.stack(split_audio, axis=0).astype(np.float64)

        return parsed_audio

    def gen_feat(self, audio: Union[str, np.ndarray]) -> torch.Tensor:
        if isinstance(audio, str):
            inp = loadWAV(
                audio,
                max_frames=self.max_frames,
                evalmode=True,
                num_eval=self.split_num,
            )
        elif isinstance(audio, np.ndarray):
            inp = self._parse_audio(
                audio, max_frames=self.max_frames, num_eval=self.split_num
            )
        else:
            raise ValueError("Invalid input type!")
        inp = torch.FloatTensor(inp).to(self.device)  # (split_num, samples)
        start = timeit.default_timer()
        emb = self.model(inp).detach().cpu()
        stop = timeit.default_timer()
        print("Ecapa forward time: %.2f s. " % (stop - start))
        emb = F.normalize(emb, p=2, dim=1)
        return emb  # (split_num, 192)

    def calc_score(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        :param emb1: (split_num, 192)
        :param emb2: (split_num, 192)
        """
        # dist (split_num, split_num)
        dist = F.cosine_similarity(
            emb1.unsqueeze(-1), emb2.unsqueeze(-1).transpose(0, 2)
        ).numpy()
        score = numpy.mean(dist)
        return score

    def calc_score_batch(self, emb1: torch.Tensor, emb2: torch.Tensor) -> numpy.ndarray:
        """
        :param emb1: (batch1, split_num, 192)
        :param emb2: (batch2, split_num, 192)
        """
        r_emb1 = emb1.reshape(emb1.size(0) * emb1.size(1), -1)
        r_emb2 = emb2.reshape(emb2.size(0) * emb1.size(1), -1)
        # dist (batch1 * split_num, batch2 * split_num)
        dist = F.cosine_similarity(
            r_emb1.unsqueeze(-1), r_emb2.unsqueeze(-1).transpose(0, 2)
        )
        # dist (batch1, split_num, batch2, split_num)
        dist = dist.reshape(emb1.size(0), emb1.size(1), emb2.size(0), emb2.size(1))
        # dist (batch1, batch2)
        dist = (
            dist.mean(dim=1, keepdim=True)
            .mean(dim=3, keepdim=True)
            .squeeze(1)
            .squeeze(2)
        )
        return dist.numpy()

    def verify(self, path1: str, path2: str) -> bool:
        emb1 = self.gen_feat(path1)
        emb2 = self.gen_feat(path2)
        score = self.calc_score(emb1, emb2)
        return score >= Args.threshold
