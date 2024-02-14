#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 声纹识别，说话人验证
@Author: chenkeming
@Date: 2024-02-14 10:36:53
"""

import timeit

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

    def gen_feat(self, path: str) -> torch.Tensor:
        inp = loadWAV(path, max_frames=300, evalmode=True)
        inp = torch.FloatTensor(inp)
        start = timeit.default_timer()
        emb = self.model(inp).detach().cpu()
        stop = timeit.default_timer()
        print("Time: %.2f s. " % (stop - start))
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def calc_score(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        dist = F.cosine_similarity(
            emb1.unsqueeze(-1), emb2.unsqueeze(-1).transpose(0, 2)
        ).numpy()
        score = numpy.mean(dist)
        return score

    def verify(self, path1: str, path2: str) -> bool:
        emb1 = self.gen_feat(path1)
        emb2 = self.gen_feat(path2)
        score = self.calc_score(emb1, emb2)
        return score >= Args.threshold
