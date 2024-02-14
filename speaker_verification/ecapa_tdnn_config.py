#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-14 10:46:28
"""


class Args:
    cfg = "speaker_verification/ecapa_tdnn/configs/ECAPA_TDNN1_step512.yaml"
    opts = None

    batch_size = None
    resume = "speaker_verification/ecapa_tdnn_weights/cnceleb_epoch_120,EER_0.7194,MinDCF_0.0747.pth"
    eval = False
    eval_model = None
    wandb = False
    note = ""

    threshold = 0.21  # 0.414
