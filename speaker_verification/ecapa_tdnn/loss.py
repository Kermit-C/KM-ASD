#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-14 09:20:42
"""

from . import lossfunction


def built_loss(cfg):
    if cfg.LOSS.NAME == "AamSoftmax":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE)

    if cfg.LOSS.NAME == "AamSoftmax_XS":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE)

    if cfg.LOSS.NAME == "AngleProto":
        loss = getattr(lossfunction, cfg.LOSS.NAME)()

    if cfg.LOSS.NAME == "AngleProtoMSE":
        loss = getattr(lossfunction, cfg.LOSS.NAME)()

    if cfg.LOSS.NAME == "MSE":
        loss = getattr(lossfunction, cfg.LOSS.NAME)()

    if cfg.LOSS.NAME == "AamSoftmaxProto":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE, w=cfg.LOSS.ALPHA)

    if cfg.LOSS.NAME == "AamSoftmaxProtoMSE":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE, w=cfg.LOSS.ALPHA)

    if cfg.LOSS.NAME == "AngleProto2":
        loss = getattr(lossfunction, cfg.LOSS.NAME)()

    if cfg.LOSS.NAME == "AamSoftmaxProto2":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE, w=cfg.LOSS.ALPHA)

    if cfg.LOSS.NAME == "AamSoftmaxCenter":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE, w=cfg.LOSS.ALPHA, lam=cfg.LOSS.LAM)

    if cfg.LOSS.NAME == "AamSoftmaxCenterA":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE, w=cfg.LOSS.ALPHA, lam=cfg.LOSS.LAM)

    if cfg.LOSS.NAME == "AamSoftmaxCenterW":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE, w=cfg.LOSS.ALPHA, lam=cfg.LOSS.LAM)

    if cfg.LOSS.NAME == "AamSoftmaxCenterWA":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES, margin=cfg.LOSS.MARGIN, scale=cfg.LOSS.SCALE, w=cfg.LOSS.ALPHA, lam=cfg.LOSS.LAM)

    if cfg.LOSS.NAME == "SAMomentumCenter":
        loss = getattr(lossfunction, cfg.LOSS.NAME)(cfg.MODEL.NEMB, cfg.LOSS.NCLASSES)

    print("loss:", cfg.LOSS.NAME)
    return loss
