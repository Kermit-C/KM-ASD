#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""


import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.loader import DataLoader
from torchvision import transforms

import active_speaker_detection.asd_config as asd_conf
import active_speaker_detection.utils.custom_transforms as ct
from active_speaker_detection.datasets.graph_dataset import GraphDataset
from active_speaker_detection.models.graph_model import get_backbone
from active_speaker_detection.optimizers.optimizer_amp import optimize_asd
from active_speaker_detection.utils.command_line import (
    get_default_arg_parser,
    unpack_command_line_args,
)
from active_speaker_detection.utils.logging import setup_optim_outputs

from .models.graph_layouts import (
    get_spatial_connection_pattern,
    get_temporal_connection_pattern,
)


def train_asd_r3d50():
    # 解析命令行参数
    command_line_args = get_default_arg_parser().parse_args()
    lr_arg, frames_per_clip, ctx_size, n_clips, strd, img_size = (
        unpack_command_line_args(command_line_args)
    )

    # 连接模式
    scp = get_spatial_connection_pattern(ctx_size, n_clips)
    tcp = get_temporal_connection_pattern(ctx_size, n_clips)

    opt_config = asd_conf.ASD_R3D_50_params
    asd_config = asd_conf.datasets

    # 数据转换
    image_size = (img_size, img_size)
    video_train_transform = transforms.Compose(
        [transforms.Resize(image_size), ct.video_train]
    )
    video_val_transform = transforms.Compose(
        [transforms.Resize(image_size), ct.video_val]
    )

    # 输出配置
    model_name = (
        "ASD_R3D_50"
        + "_clip"
        + str(frames_per_clip)
        + "_ctx"
        + str(ctx_size)
        + "_len"
        + str(n_clips)
        + "_str"
        + str(strd)
    )
    log, target_models = setup_optim_outputs(
        asd_config["models_out"], asd_config, model_name
    )

    # 创建网络并转移到GPU
    pretrain_weightds_path = opt_config["video_pretrain_weights"]
    audio_pretrain_weightds_path = opt_config["audio_pretrain_weights"]
    vfal_ecapa_pretrain_weights = opt_config["vfal_ecapa_pretrain_weights"]
    asd_net = get_backbone(
        opt_config["encoder_type"],
        pretrain_weightds_path,
        audio_pretrain_weightds_path,
        vfal_ecapa_pretrain_weights,
    )

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print("Cuda info ", has_cuda, device)
    asd_net.to(device)

    # 优化配置
    criterion = nn.CrossEntropyLoss()
    vfal_critierion = (
        losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=1.0),  # type: ignore
    )  # losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)
    optimizer = optim.Adam(asd_net.parameters(), lr=asd_config["learning_rate"])  # type: ignore
    scheduler = MultiStepLR(optimizer, milestones=[6, 8], gamma=0.1)

    # 数据路径
    video_train_path = asd_config["video_train_dir"]
    audio_train_path = asd_config["audio_train_dir"]
    video_val_path = asd_config["video_val_dir"]
    audio_val_path = asd_config["audio_val_dir"]

    # 数据加载器
    d_train = GraphDataset(
        audio_train_path,
        video_train_path,
        asd_config["csv_train_full"],
        n_clips,
        strd,
        ctx_size,
        frames_per_clip,
        scp,
        tcp,
        video_train_transform,
        do_video_augment=True,
        crop_ratio=0.95,
    )
    d_val = GraphDataset(
        audio_val_path,
        video_val_path,
        asd_config["csv_val_full"],
        n_clips,
        strd,
        ctx_size,
        frames_per_clip,
        scp,
        tcp,
        video_val_transform,
        do_video_augment=False,
    )

    dl_train = DataLoader(
        d_train,
        batch_size=opt_config["batch_size"],
        shuffle=True,
        num_workers=opt_config["threads"],
        pin_memory=True,
    )
    dl_val = DataLoader(
        d_val,
        batch_size=opt_config["batch_size"],
        shuffle=True,
        num_workers=opt_config["threads"],
        pin_memory=True,
    )

    # 优化循环
    model = optimize_asd(
        asd_net,
        dl_train,
        dl_val,
        device,
        criterion,
        vfal_critierion,
        optimizer,
        scheduler,
        num_epochs=opt_config["epochs"],
        spatial_ctx_size=ctx_size,
        time_len=n_clips,
        models_out=target_models,
        log=log,
    )
