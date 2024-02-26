#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import os

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
from active_speaker_detection.datasets.dataset_encoder import EncoderDataset
from active_speaker_detection.datasets.dataset_end2end import End2endDataset
from active_speaker_detection.datasets.dataset_graph import GraphDataset
from active_speaker_detection.datasets.gen_emb import gen_embedding
from active_speaker_detection.models.graph_model import get_backbone
from active_speaker_detection.optimizers.optimizer_encoder import optimize_encoder
from active_speaker_detection.optimizers.optimizer_end2end import optimize_end2end
from active_speaker_detection.optimizers.optimizer_graph import optimize_graph
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
        + "_vf"
        + str(1 if param_config["encoder_enable_vf"] else 0)
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
        param_config["encoder_enable_vf"],
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
    vf_critierion = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=1.0)  # type: ignore
    # losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)

    # 数据路径
    video_train_path = dataset_config["video_train_dir"]
    audio_train_path = dataset_config["audio_train_dir"]
    video_val_path = dataset_config["video_val_dir"]
    audio_val_path = dataset_config["audio_val_dir"]
    data_store_train_cache = dataset_config["data_store_train_cache"]
    data_store_val_cache = dataset_config["data_store_val_cache"]
    encoder_embedding_path = param_config["encoder_embedding_dir"]

    if stage == "end2end":
        epochs = param_config["epochs"]
        lr = param_config["learning_rate"]
        milestones = param_config["milestones"]
        gamma = param_config["gamma"]
        optimizer = optim.Adam(asd_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        d_train = End2endDataset(
            audio_train_path,
            video_train_path,
            data_store_train_cache,
            n_clips,
            strd,
            ctx_size,
            frames_per_clip,
            video_train_transform,
            do_video_augment=True,
            crop_ratio=0.95,
        )
        d_val = End2endDataset(
            audio_val_path,
            video_val_path,
            data_store_val_cache,
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
        optimize_end2end(
            asd_net,
            dl_train,
            dl_val,
            device,
            criterion,
            vf_critierion,
            optimizer,
            scheduler,
            num_epochs=epochs,
            spatial_ctx_size=ctx_size,
            time_len=n_clips,
            models_out=target_models,
            log=log,
        )

    elif stage == "graph":
        epochs = param_config["epochs"]
        lr = param_config["learning_rate"]
        milestones = param_config["milestones"]
        gamma = param_config["gamma"]
        optimizer = optim.Adam(asd_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        d_train = GraphDataset(
            os.path.join(encoder_embedding_path, "train"),
            data_store_train_cache,
            n_clips,
            strd,
            ctx_size,
        )
        d_val = GraphDataset(
            os.path.join(encoder_embedding_path, "val"),
            data_store_val_cache,
            n_clips,
            strd,
            ctx_size,
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
            optimizer,
            scheduler,
            num_epochs=epochs,
            spatial_ctx_size=ctx_size,
            time_len=n_clips,
            models_out=target_models,
            log=log,
        )

    elif stage == "encoder":
        epochs = param_config["encoder_epochs"]
        lr = param_config["encoder_learning_rate"]
        milestones = param_config["encoder_milestones"]
        gamma = param_config["encoder_gamma"]
        optimizer = optim.Adam(encoder_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        d_train = EncoderDataset(
            audio_train_path,
            video_train_path,
            data_store_train_cache,
            frames_per_clip,
            video_train_transform,
            do_video_augment=True,
            crop_ratio=0.95,
        )
        d_val = EncoderDataset(
            audio_val_path,
            video_val_path,
            data_store_val_cache,
            frames_per_clip,
            video_val_transform,
            do_video_augment=False,
        )
        dl_train = EncoderDataLoader(
            d_train,
            batch_size=param_config["encoder_batch_size"],
            shuffle=True,
            num_workers=param_config["threads"],
            pin_memory=True,
        )
        dl_val = EncoderDataLoader(
            d_val,
            batch_size=param_config["encoder_batch_size"],
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
            vf_critierion,
            optimizer,
            scheduler,
            num_epochs=epochs,
            models_out=target_models,
            log=log,
        )

    elif stage == "encoder_gen_emb":
        d_train = EncoderDataset(
            audio_train_path,
            video_train_path,
            data_store_train_cache,
            frames_per_clip,
            video_train_transform,
            do_video_augment=True,
            crop_ratio=0.95,
            eval=True,
        )
        d_val = EncoderDataset(
            audio_val_path,
            video_val_path,
            data_store_val_cache,
            frames_per_clip,
            video_val_transform,
            do_video_augment=False,
            eval=True,
        )
        dl_train = EncoderDataLoader(
            d_train,
            batch_size=param_config["encoder_batch_size"],
            shuffle=False,
            num_workers=param_config["threads"],
            pin_memory=True,
        )
        dl_val = EncoderDataLoader(
            d_val,
            batch_size=param_config["encoder_batch_size"],
            shuffle=False,
            num_workers=param_config["threads"],
            pin_memory=True,
        )
        gen_embedding(
            encoder_net,
            dl_train,
            os.path.join(encoder_embedding_path, "train"),
            device,
        )
        gen_embedding(
            encoder_net,
            dl_val,
            os.path.join(encoder_embedding_path, "val"),
            device,
        )
