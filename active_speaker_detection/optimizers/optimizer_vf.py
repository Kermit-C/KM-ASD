#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.cuda.amp.autocast_mode import autocast

from active_speaker_detection.utils.vf_util import cosine_similarity


def optimize_vf(
    model,
    dataloader_train,
    data_loader_val,
    device,
    vf_critierion,
    optimizer,
    scheduler,
    num_epochs,
    models_out=None,
    log=None,
):
    max_val_auc = 0
    for epoch in range(num_epochs):
        print()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        outs_train = _train_model_amp_avl(
            model,
            dataloader_train,
            optimizer,
            vf_critierion,
            device,
        )
        outs_val = _test_model_vf_losses(
            model,
            data_loader_val,
            vf_critierion,
            device,
        )
        scheduler.step()

        tvf_loss, train_auc = outs_train
        vvf_loss, val_auc = outs_val

        if models_out is not None and val_auc > max_val_auc:
            # 保存当前最优模型
            max_val_auc = val_auc
            model_target = os.path.join(models_out, str(epoch + 1) + ".pth")
            print("save model to ", model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.write_data_log(
                [
                    epoch + 1,
                    tvf_loss,
                    train_auc,
                    vvf_loss,
                    val_auc,
                ]
            )

    return model


def _train_model_amp_avl(
    model,
    dataloader,
    optimizer,
    vf_critierion,
    device,
):
    """训练一个 epoch 的模型"""
    model.train()

    pred_lst = []
    label_lst = []

    running_loss_vf = 0.0

    scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore

    for idx, dl in enumerate(dataloader):
        print(
            "\t Train iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_vf / (idx + 1)
            ),
            end="\r",
        )

        video_data, audio_data, _, audio_entity_idxes, video_entity_idxes, _ = dl
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        audio_entity_idxes = audio_entity_idxes.to(device)
        video_entity_idxes = video_entity_idxes.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with autocast(True):
                _, _, _, _, _, vf_a_emb, vf_v_emb = model(audio_data, video_data)
                # 音脸损失
                loss_vf = vf_critierion(
                    torch.cat([vf_a_emb, vf_v_emb], dim=0),
                    torch.cat([audio_entity_idxes, video_entity_idxes], dim=0),
                )

            scaler.scale(loss_vf).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()

        with torch.set_grad_enabled(False):
            # 选择一半进行打乱
            vf_a_idx_list = list(range(len(audio_entity_idxes)))
            vf_a_idx_list_half1 = vf_a_idx_list[: len(vf_a_idx_list) // 2]
            random.shuffle(vf_a_idx_list_half1)
            vf_a_idx_list = (
                vf_a_idx_list_half1 + vf_a_idx_list[len(vf_a_idx_list) // 2 :]
            )
            vf_a_emb = vf_a_emb[vf_a_idx_list]
            audio_entity_idxes = audio_entity_idxes[vf_a_idx_list]

            label_lst.extend(
                (audio_entity_idxes == video_entity_idxes).int().cpu().numpy().tolist()
            )
            pred_lst.extend(
                cosine_similarity(vf_a_emb, vf_v_emb).cpu().numpy().tolist()
            )

        # 统计
        running_loss_vf += loss_vf.item()
        if idx == len(dataloader) - 2:
            break

    epoch_loss_vf = running_loss_vf / len(dataloader)
    epoch_auc = roc_auc_score(label_lst, pred_lst)
    print(
        "Train Vf Loss: {:.4f}, Verify auc: {:.4f}".format(
            epoch_loss_vf,
            epoch_auc,
        )
    )
    return epoch_loss_vf, epoch_auc


def _test_model_vf_losses(
    model,
    dataloader,
    vf_critierion,
    device,
):
    """测试模型"""
    model.eval()

    pred_lst = []
    label_lst = []

    running_loss_vf = 0.0

    for idx, dl in enumerate(dataloader):
        print(
            "\t Val iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_vf / (idx + 1)
            ),
            end="\r",
        )

        video_data, audio_data, _, audio_entity_idxes, video_entity_idxes, _ = dl
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        audio_entity_idxes = audio_entity_idxes.to(device)
        video_entity_idxes = video_entity_idxes.to(device)

        with torch.set_grad_enabled(False):
            _, _, _, _, _, vf_a_emb, vf_v_emb = model(audio_data, video_data)
            # 音脸损失
            loss_vf = vf_critierion(
                torch.cat([vf_a_emb, vf_v_emb], dim=0),
                torch.cat([audio_entity_idxes, video_entity_idxes], dim=0),
            )

            # 选择一半进行打乱
            vf_a_idx_list = list(range(len(audio_entity_idxes)))
            vf_a_idx_list_half1 = vf_a_idx_list[: len(vf_a_idx_list) // 2]
            random.shuffle(vf_a_idx_list_half1)
            vf_a_idx_list = (
                vf_a_idx_list_half1 + vf_a_idx_list[len(vf_a_idx_list) // 2 :]
            )
            vf_a_emb = vf_a_emb[vf_a_idx_list]
            audio_entity_idxes = audio_entity_idxes[vf_a_idx_list]

            label_lst.extend(
                (audio_entity_idxes == video_entity_idxes).int().cpu().numpy().tolist()
            )
            pred_lst.extend(
                cosine_similarity(vf_a_emb, vf_v_emb).cpu().numpy().tolist()
            )

        # 统计
        running_loss_vf += loss_vf.item()
        if idx == len(dataloader) - 2:
            break

    epoch_loss_vf = running_loss_vf / len(dataloader)
    epoch_auc = roc_auc_score(label_lst, pred_lst)
    print(
        "Val Vf Loss: {:.4f}, Verify auc: {:.4f}".format(
            epoch_loss_vf,
            epoch_auc,
        )
    )
    return epoch_loss_vf, epoch_auc
