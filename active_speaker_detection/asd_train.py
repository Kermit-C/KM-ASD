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
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader as EncoderDataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torchvision import transforms

import active_speaker_detection.utils.custom_transforms as ct
from active_speaker_detection import asd_config
from active_speaker_detection.datasets.dataset_encoder import EncoderDataset
from active_speaker_detection.datasets.dataset_end2end import End2endDataset
from active_speaker_detection.datasets.dataset_graph import GraphDataset
from active_speaker_detection.datasets.dataset_vf import VoiceFaceDataset
from active_speaker_detection.datasets.gen_feat import gen_feature
from active_speaker_detection.models.model import get_backbone
from active_speaker_detection.optimizers.optimizer_encoder import optimize_encoder
from active_speaker_detection.optimizers.optimizer_end2end import optimize_end2end
from active_speaker_detection.optimizers.optimizer_graph import optimize_graph
from active_speaker_detection.optimizers.optimizer_vf import optimize_vf
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

    # 创建网络并转移到GPU
    encoder_enable_grad = param_config["encoder_enable_grad"]
    pretrain_weightds_path = param_config["encoder_video_pretrain_weights"]
    audio_pretrain_weightds_path = param_config["encoder_audio_pretrain_weights"]
    spatial_pretrained_weights = param_config["graph_spatial_pretrain_weights"]
    encoder_train_weights = param_config["encoder_train_weights"]
    asd_net, encoder_net = get_backbone(
        param_config["encoder_type"],
        param_config["graph_type"],
        param_config["encoder_enable_vf"],
        param_config["graph_enable_spatial"],
        encoder_enable_grad,
        pretrain_weightds_path,
        audio_pretrain_weightds_path,
        spatial_pretrained_weights,
        encoder_train_weights,
    )

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print("Cuda info ", has_cuda, device)
    asd_net.to(device)
    # asd_net = DataParallel(asd_net, device_ids=[0, 1])  # 多卡训练

    # loss
    criterion = nn.CrossEntropyLoss()
    vf_critierion = losses.MultiSimilarityLoss(alpha=2, beta=50, base=1)
    # vf_critierion = losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)

    # 数据路径
    video_train_path = dataset_config["video_train_dir"]
    audio_train_path = dataset_config["audio_train_dir"]
    video_val_path = dataset_config["video_val_dir"]
    audio_val_path = dataset_config["audio_val_dir"]
    data_store_train_cache = dataset_config["data_store_train_cache"]
    data_store_val_cache = dataset_config["data_store_val_cache"]
    encoder_feature_path = param_config["encoder_feature_dir"]

    if stage == "end2end":
        # 输出配置
        model_name = (
            stage
            + "_"
            + param_config["encoder_type"]
            + "_"
            + param_config["graph_type"]
            + str("" if encoder_enable_grad else "_pregrad0")
            + "_vf"
            + str(1 if param_config["encoder_enable_vf"] else 0)
            + "_sp"
            + str(1 if param_config["graph_enable_spatial"] else 0)
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
            dataset_config["models_out"],
            param_config,
            model_name,
            headers=[
                "epoch",
                "train_loss",
                "train_audio_loss",
                "train_video_loss",
                "train_vfal_loss",
                "train_map",
                "train_map_c",
                "val_loss",
                "val_audio_loss",
                "val_video_loss",
                "val_vfal_loss",
                "val_map",
                "val_map_c",
            ],
        )

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
            frames_per_clip,
            graph_time_steps=n_clips,
            graph_time_stride=strd,
            max_context=ctx_size,
            video_transform=video_train_transform,
            do_video_augment=True,
            crop_ratio=0.95,
        )
        d_val = End2endDataset(
            audio_val_path,
            video_val_path,
            data_store_val_cache,
            frames_per_clip,
            graph_time_steps=n_clips,
            graph_time_stride=strd,
            max_context=ctx_size,
            video_transform=video_train_transform,
            do_video_augment=False,
        )
        dl_train = GeometricDataLoader(
            d_train,
            batch_size=param_config["batch_size"],
            shuffle=True,
            num_workers=param_config["threads"],
            pin_memory=False,
        )
        dl_val = GeometricDataLoader(
            d_val,
            batch_size=param_config["batch_size"],
            shuffle=True,
            num_workers=param_config["threads"],
            pin_memory=False,
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
            accumulation_steps=param_config["accumulation_steps"],
            spatial_ctx_size=ctx_size,
            time_len=n_clips,
            models_out=target_models,
            log=log,
        )

    elif stage == "graph":
        # 输出配置
        model_name = (
            stage
            + "_"
            + param_config["encoder_type"]
            + "_"
            + param_config["graph_type"]
            + "_vf"
            + str(1 if param_config["encoder_enable_vf"] else 0)
            + "_sp"
            + str(1 if param_config["graph_enable_spatial"] else 0)
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
            dataset_config["models_out"],
            param_config,
            model_name,
            headers=[
                "epoch",
                "train_loss",
                "train_audio_loss",
                "train_video_loss",
                "train_map",
                "train_map_c",
                "val_loss",
                "val_audio_loss",
                "val_video_loss",
                "val_map",
                "val_map_c",
            ],
        )

        epochs = param_config["graph_epochs"]
        lr = param_config["graph_learning_rate"]
        milestones = param_config["graph_milestones"]
        gamma = param_config["graph_gamma"]
        optimizer = optim.Adam(asd_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        d_train = GraphDataset(
            os.path.join(encoder_feature_path, "train"),
            data_store_train_cache,
            graph_time_steps=n_clips,
            graph_time_stride=strd,
            max_context=ctx_size,
        )
        d_val = GraphDataset(
            os.path.join(encoder_feature_path, "val"),
            data_store_val_cache,
            graph_time_steps=n_clips,
            graph_time_stride=strd,
            max_context=ctx_size,
        )
        dl_train = GeometricDataLoader(
            d_train,
            batch_size=param_config["graph_batch_size"],
            shuffle=True,
            num_workers=param_config["graph_threads"],
            pin_memory=False,  # 用数据集自己实现的 Cache
        )
        dl_val = GeometricDataLoader(
            d_val,
            batch_size=param_config["graph_batch_size"],
            shuffle=True,
            num_workers=param_config["graph_threads"],
            pin_memory=False,  # 用数据集自己实现的 Cache
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
        # 输出配置
        model_name = (
            stage
            + "_"
            + param_config["encoder_type"]
            + str("" if encoder_enable_grad else "_pregrad0")
            + "_vf"
            + str(1 if param_config["encoder_enable_vf"] else 0)
            + "_clip"
            + str(frames_per_clip)
        )
        log, target_models = setup_optim_outputs(
            dataset_config["models_out"],
            param_config,
            model_name,
            headers=[
                "epoch",
                "train_loss",
                "train_audio_loss",
                "train_video_loss",
                "train_vfal_loss",
                "train_map",
                "val_loss",
                "val_audio_loss",
                "val_video_loss",
                "val_vfal_loss",
                "val_map",
            ],
        )

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
            num_workers=param_config["encoder_threads"],
            pin_memory=True,
        )
        dl_val = EncoderDataLoader(
            d_val,
            batch_size=param_config["encoder_batch_size"],
            shuffle=True,
            num_workers=param_config["encoder_threads"],
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
            a_weight=0.2,
            v_weight=0.5,
            vf_weight=0.5,
            models_out=target_models,
            log=log,
        )

    elif stage == "encoder_vf":
        # 输出配置
        model_name = (
            stage
            + "_"
            + param_config["encoder_type"]
            + str("" if encoder_enable_grad else "_pregrad0")
            + "_vf"
            + str(1 if param_config["encoder_enable_vf"] else 0)
            + "_clip"
            + str(frames_per_clip)
        )
        log, target_models = setup_optim_outputs(
            dataset_config["models_out"],
            param_config,
            model_name,
            headers=[
                "epoch",
                "train_vfal_loss",
                "train_auc",
                "val_vfal_loss",
                "val_auc",
            ],
        )

        epochs = param_config["encoder_epochs"]
        lr = param_config["encoder_learning_rate"]
        milestones = param_config["encoder_milestones"]
        gamma = param_config["encoder_gamma"]
        optimizer = optim.Adam(encoder_net.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        d_train = VoiceFaceDataset(
            audio_train_path,
            video_train_path,
            data_store_train_cache,
            frames_per_clip,
            video_train_transform,
            do_video_augment=True,
            crop_ratio=0.95,
        )
        d_val = VoiceFaceDataset(
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
            num_workers=param_config["encoder_threads"],
            pin_memory=True,
        )
        dl_val = EncoderDataLoader(
            d_val,
            batch_size=param_config["encoder_batch_size"],
            shuffle=True,
            num_workers=param_config["encoder_threads"],
            pin_memory=True,
        )
        optimize_vf(
            encoder_net,
            dl_train,
            dl_val,
            device,
            vf_critierion,
            optimizer,
            scheduler,
            num_epochs=epochs,
            models_out=target_models,
            log=log,
        )

    elif stage == "encoder_gen_feat":
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
            num_workers=param_config["encoder_threads"],
            pin_memory=True,
        )
        dl_val = EncoderDataLoader(
            d_val,
            batch_size=param_config["encoder_batch_size"],
            shuffle=False,
            num_workers=param_config["encoder_threads"],
            pin_memory=True,
        )
        gen_feature(
            encoder_net,
            dl_train,
            os.path.join(encoder_feature_path, "train"),
            device,
        )
        gen_feature(
            encoder_net,
            dl_val,
            os.path.join(encoder_feature_path, "val"),
            device,
        )
