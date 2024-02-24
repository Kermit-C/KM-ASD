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
from torch.utils.data import DataLoader as EncoderDataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torchvision import transforms

import active_speaker_detection.utils.custom_transforms as ct
from active_speaker_detection import asd_config
from active_speaker_detection.datasets.encoder_dataset import EncoderDataset
from active_speaker_detection.datasets.graph_dataset import GraphDataset
from active_speaker_detection.models.graph_model import get_backbone
from active_speaker_detection.optimizers.encoder_optimizer import optimize_encoder
from active_speaker_detection.optimizers.graph_optimizer import optimize_graph
from active_speaker_detection.utils.command_line import (
    get_default_arg_parser,
    unpack_command_line_args,
)
from active_speaker_detection.utils.logging import setup_optim_outputs


def train():
    # 解析命令行参数
    command_line_args = get_default_arg_parser().parse_args()
    name, stage, frames_per_clip, ctx_size, n_clips, strd, img_size = (
        unpack_command_line_args(command_line_args)
    )

    param_config = next(filter(lambda x: x["name"] == name, asd_config.train_params))
    dataset_config = asd_config.datasets

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
        param_config["encoder_type"]
        + "_stage_"
        + stage
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
        dataset_config["models_out"], param_config, model_name
    )

    # 创建网络并转移到GPU
    pretrain_weightds_path = param_config["encoder_video_pretrain_weights"]
    audio_pretrain_weightds_path = param_config["encoder_audio_pretrain_weights"]
    vfal_ecapa_pretrain_weights = param_config["encoder_vfal_ecapa_pretrain_weights"]
    encoder_train_weights = param_config["encoder_train_weights"]
    asd_net, encoder_net = get_backbone(
        param_config["encoder_type"],
        pretrain_weightds_path,
        audio_pretrain_weightds_path,
        vfal_ecapa_pretrain_weights,
        encoder_train_weights,
    )

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print("Cuda info ", has_cuda, device)
    asd_net.to(device)

    # loss
    criterion = nn.CrossEntropyLoss()
    vfal_critierion = (
        losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=1.0),  # type: ignore
    )  # losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)

    # 数据路径
    video_train_path = dataset_config["video_train_dir"]
    audio_train_path = dataset_config["audio_train_dir"]
    video_val_path = dataset_config["video_val_dir"]
    audio_val_path = dataset_config["audio_val_dir"]

    if stage == "graph" or stage == "end2end":
        epochs = param_config["epochs"]
        lr = param_config["learning_rate"]
        milestones = param_config["milestones"]
        gamma = param_config["gamma"]
        optimizer = optim.Adam(asd_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        d_train = GraphDataset(
            audio_train_path,
            video_train_path,
            dataset_config["csv_train_full"],
            n_clips,
            strd,
            ctx_size,
            frames_per_clip,
            video_train_transform,
            do_video_augment=True,
            crop_ratio=0.95,
        )
        d_val = GraphDataset(
            audio_val_path,
            video_val_path,
            dataset_config["csv_val_full"],
            n_clips,
            strd,
            ctx_size,
            frames_per_clip,
            video_val_transform,
            do_video_augment=False,
        )
        dl_train = GeometricDataLoader(
            d_train,
            batch_size=param_config["batch_size"],
            shuffle=True,
            num_workers=param_config["threads"],
            pin_memory=True,
        )
        dl_val = GeometricDataLoader(
            d_val,
            batch_size=param_config["batch_size"],
            shuffle=True,
            num_workers=param_config["threads"],
            pin_memory=True,
        )
        optimize_graph(
            asd_net,
            dl_train,
            dl_val,
            device,
            criterion,
            vfal_critierion,
            optimizer,
            scheduler,
            num_epochs=epochs,
            spatial_ctx_size=ctx_size,
            time_len=n_clips,
            models_out=target_models,
            log=log,
        )

    else:
        epochs = param_config["encoder_epochs"]
        lr = param_config["encoder_learning_rate"]
        milestones = param_config["encoder_milestones"]
        gamma = param_config["encoder_gamma"]
        optimizer = optim.Adam(asd_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        d_train = EncoderDataset(
            audio_train_path,
            video_train_path,
            dataset_config["csv_train_full"],
            frames_per_clip,
            video_train_transform,
            do_video_augment=True,
            crop_ratio=0.95,
        )
        d_val = EncoderDataset(
            audio_val_path,
            video_val_path,
            dataset_config["csv_val_full"],
            frames_per_clip,
            video_val_transform,
            do_video_augment=False,
        )
        dl_train = EncoderDataLoader(
            d_train,
            batch_size=param_config["batch_size"],
            shuffle=True,
            num_workers=param_config["threads"],
            pin_memory=True,
        )
        dl_val = EncoderDataLoader(
            d_val,
            batch_size=param_config["batch_size"],
            shuffle=True,
            num_workers=param_config["threads"],
            pin_memory=True,
        )
        optimize_encoder(
            encoder_net,
            dl_train,
            dl_val,
            device,
            criterion,
            optimizer,
            scheduler,
            num_epochs=epochs,
            models_out=target_models,
            log=log,
        )
