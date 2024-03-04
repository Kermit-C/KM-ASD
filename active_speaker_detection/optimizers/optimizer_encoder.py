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


def optimize_encoder(
    model,
    dataloader_train,
    data_loader_val,
    device,
    criterion,
    vf_critierion,
    optimizer,
    scheduler,
    num_epochs,
    a_weight=0.2,
    v_weight=0.5,
    vf_weight=0.1,
    models_out=None,
    log=None,
):
    if models_out is not None and os.path.exists(os.path.join(models_out, "last.ckpt")):
        # 加载上次训练的状态
        checkpoint = torch.load(os.path.join(models_out, "last.ckpt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        max_val_ap = checkpoint["max_val_ap"]
        start_epoch = checkpoint["epoch"]
        print("load checkpoint from ", os.path.join(models_out, "last.ckpt"))
    else:
        start_epoch = 0
        max_val_ap = 0

    for epoch in range(start_epoch, num_epochs):
        print()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        outs_train = _train_model_amp_avl(
            model,
            dataloader_train,
            optimizer,
            criterion,
            vf_critierion,
            device,
            a_weight,
            v_weight,
            vf_weight,
        )
        outs_val = _test_model_encoder_losses(
            model,
            data_loader_val,
            criterion,
            vf_critierion,
            device,
            a_weight,
            v_weight,
            vf_weight,
        )
        scheduler.step()

        train_loss, ta_loss, tv_loss, tvf_loss, train_ap = outs_train
        val_loss, va_loss, vv_loss, vvf_loss, val_ap = outs_val

        if models_out is not None and val_ap > max_val_ap:
            # 保存当前最优模型
            max_val_ap = val_ap
            model_target = os.path.join(models_out, str(epoch + 1) + ".pth")
            print("save model to ", model_target)
            torch.save(model.state_dict(), model_target)

        if models_out is not None:
            # 保存当前训练进度，包含 epoch 和 optimizer 的状态
            model_target = os.path.join(models_out, "last.ckpt")
            print("save checkpoint to ", model_target)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "max_val_ap": max_val_ap,
                },
                model_target,
            )

        if log is not None:
            log.write_data_log(
                [
                    epoch + 1,
                    train_loss,
                    ta_loss,
                    tv_loss,
                    tvf_loss,
                    train_ap,
                    val_loss,
                    va_loss,
                    vv_loss,
                    vvf_loss,
                    val_ap,
                ]
            )

    return model


def _train_model_amp_avl(
    model,
    dataloader,
    optimizer,
    criterion: nn.modules.loss._Loss,
    vf_critierion,
    device,
    a_weight=0.2,
    v_weight=0.5,
    vf_weight=0.3,
):
    """训练一个 epoch 的模型，返回图的损失和音频视频的辅助损失"""
    model.train()
    softmax_layer = torch.nn.Softmax(dim=1)

    pred_lst = []
    label_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_loss_vf = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore

    for idx, dl in enumerate(dataloader):
        print(
            "\t Train iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_g / (idx + 1)
            ),
            end="\r",
        )

        (
            video_data,
            audio_data,
            _,
            target,
            target_a,
            audio_entity_idxes,
            video_entity_idxes,
            _,
            _,
        ) = dl
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        target = target.to(device)
        target_a = target_a.to(device)
        audio_entity_idxes = audio_entity_idxes.to(device)
        video_entity_idxes = video_entity_idxes.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with autocast(True):
                _, _, audio_out, video_out, av_out, vf_a_emb, vf_v_emb = model(
                    audio_data, video_data
                )
                # 单独音频和视频的损失
                loss_a: torch.Tensor = criterion(audio_out, target_a)
                loss_v: torch.Tensor = criterion(video_out, target)
                loss_av: torch.Tensor = criterion(av_out, target)
                if vf_a_emb is not None and vf_v_emb is not None:
                    # 音脸损失
                    loss_vf = vf_critierion(
                        torch.cat([vf_a_emb, vf_v_emb], dim=0),
                        torch.cat([audio_entity_idxes, video_entity_idxes], dim=0),
                    )
                    loss = (
                        vf_weight * loss_vf
                        + a_weight * loss_a
                        + v_weight * loss_v
                        + loss_av
                    )
                else:
                    loss = a_weight * loss_a + v_weight * loss_v + loss_av

            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()

        with torch.set_grad_enabled(False):
            label_lst.extend(target.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(av_out).cpu().numpy()[:, 1].tolist())

        # 统计
        running_loss_g += loss.item()
        running_loss_a += loss_a.item()
        running_loss_v += loss_v.item()
        running_loss_vf += (
            loss_vf.item() if vf_a_emb is not None and vf_v_emb is not None else 0
        )
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_loss_vf = running_loss_vf / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print(
        "Train Total Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, Vf Loss: {:.4f}, mAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_loss_vf,
            epoch_ap,
        )
    )
    return epoch_loss_g, epoch_loss_a, epoch_loss_v, epoch_loss_vf, epoch_ap


def _test_model_encoder_losses(
    model,
    dataloader,
    criterion: nn.modules.loss._Loss,
    vf_critierion,
    device,
    a_weight=0.2,
    v_weight=0.5,
    vf_weight=0.3,
):
    """测试模型，返回图的损失和音频视频的辅助损失"""
    model.eval()
    softmax_layer = torch.nn.Softmax(dim=1)

    pred_lst = []
    label_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_loss_vf = 0.0

    for idx, dl in enumerate(dataloader):
        print(
            "\t Val iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_g / (idx + 1)
            ),
            end="\r",
        )

        (
            video_data,
            audio_data,
            _,
            target,
            target_a,
            audio_entity_idxes,
            video_entity_idxes,
            _,
            _,
        ) = dl
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        target = target.to(device)
        target_a = target_a.to(device)
        audio_entity_idxes = audio_entity_idxes.to(device)
        video_entity_idxes = video_entity_idxes.to(device)

        with torch.set_grad_enabled(False):
            _, _, audio_out, video_out, av_out, vf_a_emb, vf_v_emb = model(
                audio_data, video_data
            )
            # 单独音频和视频的损失
            loss_a: torch.Tensor = criterion(audio_out, target_a)
            loss_v: torch.Tensor = criterion(video_out, target)
            loss_av: torch.Tensor = criterion(av_out, target)
            if vf_a_emb is not None and vf_v_emb is not None:
                # 音脸损失
                loss_vf = vf_critierion(
                    torch.cat([vf_a_emb, vf_v_emb], dim=0),
                    torch.cat([audio_entity_idxes, video_entity_idxes], dim=0),
                )
                loss = (
                    vf_weight * loss_vf
                    + a_weight * loss_a
                    + v_weight * loss_v
                    + loss_av
                )
            else:
                loss = a_weight * loss_a + v_weight * loss_v + loss_av

            label_lst.extend(target.cpu().numpy().tolist())
            pred_lst.extend(softmax_layer(av_out).cpu().numpy()[:, 1].tolist())

        # 统计
        running_loss_g += loss.item()
        running_loss_a += loss_a.item()
        running_loss_v += loss_v.item()
        running_loss_vf += (
            loss_vf.item() if vf_a_emb is not None and vf_v_emb is not None else 0
        )
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_loss_vf = running_loss_vf / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    print(
        "Val Total Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, Vf Loss: {:.4f}, mAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_loss_vf,
            epoch_ap,
        )
    )
    return (
        epoch_loss_g,
        epoch_loss_a,
        epoch_loss_v,
        epoch_loss_vf,
        epoch_ap,
    )
