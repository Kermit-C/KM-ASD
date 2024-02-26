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
from sklearn.metrics import average_precision_score
from torch.cuda.amp.autocast_mode import autocast

from active_speaker_detection.models.graph_layouts import (
    generate_av_mask,
    generate_temporal_video_center_mask,
    generate_temporal_video_mask,
)


def optimize_graph(
    model,
    dataloader_train,
    data_loader_val,
    device,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    spatial_ctx_size,
    time_len,  # 图的时间步数
    models_out=None,
    log=None,
):
    max_val_ap = 0
    for epoch in range(num_epochs):
        print()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        outs_train = _train_model_amp_avl(
            model,
            dataloader_train,
            optimizer,
            criterion,
            device,
            spatial_ctx_size,
            time_len,
        )
        outs_val = _test_model_graph_losses(
            model,
            data_loader_val,
            criterion,
            device,
            spatial_ctx_size,
            time_len,
        )
        scheduler.step()

        train_loss, ta_loss, tv_loss, train_ap = outs_train
        val_loss, va_loss, vv_loss, val_ap, val_tap, val_cap = outs_val

        if models_out is not None and val_ap > max_val_ap:
            # 保存当前最优模型
            max_val_ap = val_ap
            model_target = os.path.join(models_out, str(epoch + 1) + ".pth")
            print("save model to ", model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog(
                [
                    epoch + 1,
                    train_loss,
                    ta_loss,
                    tv_loss,
                    0,
                    train_ap,
                    val_loss,
                    va_loss,
                    vv_loss,
                    0,
                    val_ap,
                    val_tap,
                    val_cap,
                ]
            )

    return model


def _train_model_amp_avl(
    model,
    dataloader,
    optimizer,
    criterion: nn.modules.loss._Loss,
    device,
    ctx_size,
    time_len: int,  # 图的时间步数
):
    """训练一个 epoch 的模型，返回图的损失和音频视频的辅助损失"""
    model.train()
    softmax_layer = torch.nn.Softmax(dim=1)

    pred_lst = []
    label_lst = []

    pred_time_lst = []
    label_time_lst = []

    pred_center_lst = []
    label_center_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore

    for idx, dl in enumerate(dataloader):
        print(
            "\t Train iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_g / (idx + 1)
            ),
            end="\r",
        )

        graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y
        entities = graph_data.y2

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # 生成掩码, 用于计算辅助损失, 包括单独的音频和视频损失
            audio_mask, video_mask = generate_av_mask(ctx_size, graph_data.x.size(0))
            # 生成单人时序上的掩码，仅用来计算单人时序上的损失，仅用于评估
            temporal_video_mask = generate_temporal_video_mask(
                ctx_size, graph_data.x.size(0)
            )
            # 生成中心帧的掩码，仅用来计算中心帧的损失，仅用于评估
            center_mask = generate_temporal_video_center_mask(
                ctx_size, graph_data.x.size(0), time_len
            )

            with autocast(True):
                outputs, _, _, _, _ = model(graph_data, ctx_size)
                # 图的损失
                loss: torch.Tensor = criterion(outputs, targets)

            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()

        with torch.set_grad_enabled(False):
            label_lst.extend(targets[video_mask].cpu().numpy().tolist())
            pred_lst.extend(
                softmax_layer(outputs[video_mask]).cpu().numpy()[:, 1].tolist()
            )

            label_time_lst.extend(targets[temporal_video_mask].cpu().numpy().tolist())
            pred_time_lst.extend(
                softmax_layer(outputs[temporal_video_mask]).cpu().numpy()[:, 1].tolist()
            )

            label_center_lst.extend(targets[center_mask].cpu().numpy().tolist())
            pred_center_lst.extend(
                softmax_layer(outputs[center_mask]).cpu().numpy()[:, 1].tolist()
            )

        # 统计
        running_loss_g += loss.item()
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_time_ap = average_precision_score(label_time_lst, pred_time_lst)
    epoch_center_ap = average_precision_score(label_center_lst, pred_center_lst)
    print(
        "Train Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, VmAP: {:.4f}, TVmAP: {:.4f}, CVmAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_ap,
            epoch_time_ap,
            epoch_center_ap,
        )
    )
    return epoch_loss_g, epoch_loss_a, epoch_loss_v, epoch_ap


def _test_model_graph_losses(
    model,
    dataloader,
    criterion: nn.modules.loss._Loss,
    device,
    ctx_size,
    time_len,  # 图的时间步数
):
    """测试模型，返回图的损失和音频视频的辅助损失"""
    model.eval()
    softmax_layer = torch.nn.Softmax(dim=1)

    pred_lst = []
    label_lst = []

    pred_time_lst = []
    label_time_lst = []

    pred_center_lst = []
    label_center_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0

    audio_size, vfal_size = dataloader.dataset.get_audio_size()

    for idx, dl in enumerate(dataloader):
        print(
            "\t Val iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_g / (idx + 1)
            ),
            end="\r",
        )

        graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y
        entities = graph_data.y2

        with torch.set_grad_enabled(False):
            audio_mask, video_mask = generate_av_mask(ctx_size, graph_data.x.size(0))
            temporal_video_mask = generate_temporal_video_mask(
                ctx_size, graph_data.x.size(0)
            )
            center_mask = generate_temporal_video_center_mask(
                ctx_size, graph_data.x.size(0), time_len
            )

            outputs, _, _, _, _ = model(graph_data, ctx_size, audio_size, vfal_size)
            loss = criterion(outputs, targets)

            label_lst.extend(targets[video_mask].cpu().numpy().tolist())
            pred_lst.extend(
                softmax_layer(outputs[video_mask]).cpu().numpy()[:, 1].tolist()
            )

            label_time_lst.extend(targets[temporal_video_mask].cpu().numpy().tolist())
            pred_time_lst.extend(
                softmax_layer(outputs[temporal_video_mask]).cpu().numpy()[:, 1].tolist()
            )

            label_center_lst.extend(targets[center_mask].cpu().numpy().tolist())
            pred_center_lst.extend(
                softmax_layer(outputs[center_mask]).cpu().numpy()[:, 1].tolist()
            )

        # 统计
        running_loss_g += loss.item()
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_time_ap = average_precision_score(label_time_lst, pred_time_lst)
    epoch_center_ap = average_precision_score(label_center_lst, pred_center_lst)
    print(
        "Val Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, VmAP: {:.4f}, TVmAP: {:.4f}, CVmAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_ap,
            epoch_time_ap,
            epoch_center_ap,
        )
    )
    return (
        epoch_loss_g,
        epoch_loss_a,
        epoch_loss_v,
        epoch_ap,
        epoch_time_ap,
        epoch_center_ap,
    )
