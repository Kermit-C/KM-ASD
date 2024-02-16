#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-05 23:05:15
"""

import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses

from .models import graph_models as graph_model

ASD_R3D_18_inputs = {
    # 输入文件
    "csv_train_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_train_augmented.csv",
    "csv_val_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_val_augmented.csv",
    "csv_test_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_test_augmented.csv",
    # 数据配置
    "audio_train_dir": "/ssd1/ckm2/instance_wavs_time_train/",
    "video_train_dir": "/ssd1/ckm2/instance_crops_time_train/",
    "audio_val_dir": "/ssd1/ckm2/instance_wavs_time_val/",
    "video_val_dir": "/ssd1/ckm2/instance_crops_time_val/",
    "models_out": "./active_speaker_detection/results",  # 保存目录
    # 预训练权重
    "audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
    "video_pretrain_weights": "/hdd1/ckm/pretrain-model/3D-ResNets-PyTorch/results/r3d18_K_200ep.pth",
    "vfal_ecapa_pretrain_weights": "/hdd1/ckm/pretrain-model/ECAPA_TDNN/results/ecapa_acc0.9854.pkl",
}

ASD_R3D_50_inputs = {
    # 输入文件
    "csv_train_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_train_augmented.csv",
    "csv_val_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_val_augmented.csv",
    "csv_test_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_test_augmented.csv",
    # 数据配置
    "audio_train_dir": "/ssd1/ckm2/instance_wavs_time_train/",
    "video_train_dir": "/ssd1/ckm2/instance_crops_time_train/",
    "audio_val_dir": "/ssd1/ckm2/instance_wavs_time_val/",
    "video_val_dir": "/ssd1/ckm2/instance_crops_time_val/",
    "models_out": "./active_speaker_detection/results",  # 保存目录
    # 预训练权重
    "audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
    "video_pretrain_weights": "/hdd1/ckm/pretrain-model/3D-ResNets-PyTorch/results/r3d50_K_200ep.pth",
    "vfal_ecapa_pretrain_weights": "/hdd1/ckm/pretrain-model/ECAPA_TDNN/results/ecapa_acc0.9854.pkl",
}


ASD_R3D_18_4lvl_params = {
    # 网络架构
    "backbone": graph_model.R3D18_4lvlGCN,
    # 优化配置
    "optimizer": optim.Adam,
    "criterion": nn.CrossEntropyLoss(),
    "vfal_criterion": losses.LiftedStructureLoss(neg_margin=1, pos_margin=0),
    "learning_rate": 3e-4,
    "epochs": 15,
    "gamma": 0.1,
    # 批次配置
    "batch_size": 17,
    "threads": 8,
}


ASD_R3D_50_4lvl_params = {
    # 网络架构
    "backbone": graph_model.R3D50_4lvlGCN,
    # 优化配置
    "optimizer": optim.Adam,
    "criterion": nn.CrossEntropyLoss(),
    "vfal_criterion": losses.LiftedStructureLoss(neg_margin=1, pos_margin=0),
    "learning_rate": 3e-4,
    "epochs": 15,
    "gamma": 0.1,
    # 批次配置
    "batch_size": 17,
    "threads": 8,
}
