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


def optimize_asd(
    model,
    dataloader_train,
    data_loader_val,
    device,
    criterion,
    vfal_critierion,
    optimizer,
    scheduler,
    num_epochs,
    spatial_ctx_size,
    time_len,  # 图的时间步数
    a_weight=0.2,
    v_weight=0.5,
    vfal_weight=0.3,
    models_out=None,
    log=None,
):

    for epoch in range(num_epochs):
        print()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        outs_train = _train_model_amp_avl(
            model,
            dataloader_train,
            optimizer,
            criterion,
            vfal_critierion,
            device,
            spatial_ctx_size,
            time_len,
            a_weight,
            v_weight,
            vfal_weight,
        )
        outs_val = _test_model_graph_losses(
            model,
            data_loader_val,
            criterion,
            vfal_critierion,
            device,
            spatial_ctx_size,
            time_len,
        )
        scheduler.step()

        train_loss, ta_loss, tv_loss, tvfal_loss, train_ap = outs_train
        val_loss, va_loss, vv_loss, vvfal_loss, val_ap, val_tap, val_cap = outs_val

        if models_out is not None and epoch > num_epochs - 10:
            # 保存最后 10 个 epoch 的模型
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
                    tvfal_loss,
                    train_ap,
                    val_loss,
                    va_loss,
                    vv_loss,
                    vvfal_loss,
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
    vfal_critierion: nn.modules.loss._Loss,
    device,
    ctx_size,
    time_len: int,  # 图的时间步数
    a_weight,
    v_weight,
    vfal_weight,
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
    running_loss_vfal = 0.0

    audio_size, vfal_size = dataloader.dataset.get_audio_size()
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore

    # Iterate over data
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
            # TODO inneficient here
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
                outputs, audio_out, video_out, vfal_a_feats, vfal_v_feats = model(
                    graph_data, ctx_size, audio_size, vfal_size
                )
                # 单独音频和视频的损失
                aux_loss_a: torch.Tensor = criterion(audio_out, targets[audio_mask])
                aux_loss_v: torch.Tensor = criterion(video_out, targets[video_mask])
                # aux_loss_vfal: torch.Tensor = vfal_critierion(
                #     torch.cat([vfal_a_feats, vfal_v_feats], dim=0),
                #     torch.cat([entities[audio_mask], entities[video_mask]], dim=0),
                # )
                # 图的损失
                loss_graph: torch.Tensor = criterion(outputs, targets)
                loss = (
                    a_weight * aux_loss_a
                    + v_weight * aux_loss_v
                    # + vfal_weight * aux_loss_vfal
                    + loss_graph
                )

            optimizer.zero_grad()  # 重置梯度，不加会爆显存
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

        # statistics
        running_loss_g += loss_graph.item()
        running_loss_a += aux_loss_a.item()
        running_loss_v += aux_loss_v.item()
        # running_loss_vfal += aux_loss_vfal.item()
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_loss_vfal = running_loss_vfal / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_time_ap = average_precision_score(label_time_lst, pred_time_lst)
    epoch_center_ap = average_precision_score(label_center_lst, pred_center_lst)
    print(
        "Train Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, Vfal Loss: {:.4f}, VmAP: {:.4f}, TVmAP: {:.4f}, CVmAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_loss_vfal,
            epoch_ap,
            epoch_time_ap,
            epoch_center_ap,
        )
    )
    return epoch_loss_g, epoch_loss_a, epoch_loss_v, epoch_loss_vfal, epoch_ap


def _test_model_graph_losses(
    model,
    dataloader,
    criterion: nn.modules.loss._Loss,
    vfal_critierion: nn.modules.loss._Loss,
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
    running_loss_vfal = 0.0

    audio_size, vfal_size = dataloader.dataset.get_audio_size()

    # Iterate over data
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
            # TODO inneficient here
            audio_mask, video_mask = generate_av_mask(ctx_size, graph_data.x.size(0))
            temporal_video_mask = generate_temporal_video_mask(
                ctx_size, graph_data.x.size(0)
            )
            center_mask = generate_temporal_video_center_mask(
                ctx_size, graph_data.x.size(0), time_len
            )

            outputs, audio_out, video_out, vfal_a_feats, vfal_v_feats = model(
                graph_data, ctx_size, audio_size, vfal_size
            )
            loss_graph = criterion(outputs, targets)
            aux_loss_a = criterion(audio_out, targets[audio_mask])
            aux_loss_v = criterion(video_out, targets[video_mask])
            # aux_loss_vfal: torch.Tensor = vfal_critierion(
            #     torch.cat([vfal_a_feats, vfal_v_feats], dim=0),
            #     torch.cat(
            #         [entities[audio_mask], entities[video_mask]], dim=0
            #     ).squeeze(),
            # )

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

        # statistics
        running_loss_g += loss_graph.item()
        running_loss_a += aux_loss_a.item()
        running_loss_v += aux_loss_v.item()
        # running_loss_vfal += aux_loss_vfal.item()

        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_loss_vfal = running_loss_vfal / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_time_ap = average_precision_score(label_time_lst, pred_time_lst)
    epoch_center_ap = average_precision_score(label_center_lst, pred_center_lst)
    print(
        "Val Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, Vfal Loss: {:.4f}, VmAP: {:.4f}, TVmAP: {:.4f}, CVmAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_loss_vfal,
            epoch_ap,
            epoch_time_ap,
            epoch_center_ap,
        )
    )
    return (
        epoch_loss_g,
        epoch_loss_a,
        epoch_loss_v,
        epoch_loss_vfal,
        epoch_ap,
        epoch_time_ap,
        epoch_center_ap,
    )
