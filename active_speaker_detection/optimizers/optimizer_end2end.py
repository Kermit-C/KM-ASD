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


def optimize_end2end(
    model,
    dataloader_train,
    data_loader_val,
    device,
    criterion,
    vf_critierion,
    optimizer,
    scheduler,
    num_epochs,
    accumulation_steps,
    spatial_ctx_size,
    time_len,  # 图的时间步数
    a_weight=0.2,
    v_weight=0.5,
    vfal_weight=0.1,
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
            accumulation_steps,
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
            vf_critierion,
            device,
            spatial_ctx_size,
            time_len,
        )
        scheduler.step()

        (
            train_loss,
            ta_loss,
            tv_loss,
            tvfal_loss,
            train_ap,
            train_ap_center_node,
            train_ap_last_node,
        ) = outs_train
        (
            val_loss,
            va_loss,
            vv_loss,
            vvfal_loss,
            val_ap,
            val_ap_center_node,
            val_ap_last_node,
        ) = outs_val

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
                    tvfal_loss,
                    train_ap,
                    train_ap_center_node,
                    train_ap_last_node,
                    val_loss,
                    va_loss,
                    vv_loss,
                    vvfal_loss,
                    val_ap,
                    val_ap_center_node,
                    val_ap_last_node,
                ]
            )

    return model


def _train_model_amp_avl(
    model,
    dataloader,
    optimizer,
    criterion: nn.modules.loss._Loss,
    vf_critierion: nn.modules.loss._Loss,
    device,
    accumulation_steps,
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
    center_node_pred_lst = []
    center_node_label_lst = []
    last_node_pred_lst = []
    last_node_label_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_loss_vf = 0.0

    audio_size, _ = dataloader.dataset.get_audio_size()
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore
    optimizer.zero_grad()

    for idx, dl in enumerate(dataloader):
        print(
            "\t Train iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_g / (idx + 1)
            ),
            end="\r",
        )

        graph_data = dl
        graph_data = graph_data.to(device)

        center_node_mask = []
        last_node_mask = []
        for c_mask, l_mask in zip(
            graph_data.center_node_mask, graph_data.last_node_mask
        ):
            center_node_mask += c_mask
            last_node_mask += l_mask
        audio_node_mask = []
        for mask in graph_data.audio_node_mask:
            audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        targets_a = graph_data.y[:, 0]
        targets_v = graph_data.y[:, 1]
        targets_g = targets_v.clone()
        targets_g[audio_node_mask] = targets_a[audio_node_mask]
        entities_a = graph_data.y[:, 2]
        entities_v = graph_data.y[:, 3]

        with torch.set_grad_enabled(True):
            with autocast(True):
                outputs, audio_out, video_out, vf_a_emb, vf_v_emb = model(
                    graph_data, audio_size
                )
                # 单独音频和视频的损失
                aux_loss_a = criterion(audio_out, targets_a[audio_node_mask])
                aux_loss_v = criterion(video_out, targets_v[video_node_mask])
                if vf_a_emb is not None and vf_v_emb is not None:
                    # 音脸损失
                    aux_loss_vf: torch.Tensor = vf_critierion(
                        torch.cat([vf_a_emb, vf_v_emb], dim=0),
                        torch.cat([entities_a, entities_v], dim=0),
                    )
                    # 图的损失
                    loss_graph: torch.Tensor = criterion(outputs, targets_g)
                    loss = (
                        a_weight * aux_loss_a
                        + v_weight * aux_loss_v
                        + vfal_weight * aux_loss_vf
                        + loss_graph
                    )
                else:
                    # 图的损失
                    loss_graph: torch.Tensor = criterion(outputs, targets_g)
                    loss = a_weight * aux_loss_a + v_weight * aux_loss_v + loss_graph

            loss /= accumulation_steps
            scaler.scale(loss).backward()  # type: ignore
            if (idx + 1) % accumulation_steps == 0 or idx == len(dataloader) - 1:
                # 累计 accumulation_steps 个 batch 的梯度，然后更新
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()

        with torch.set_grad_enabled(False):
            label_lst.extend(targets_g[video_node_mask].cpu().numpy().tolist())
            pred_lst.extend(
                softmax_layer(outputs[video_node_mask]).cpu().numpy()[:, 1].tolist()
            )

            center_node_label_lst.extend(
                targets_g[[v and c for v, c in zip(video_node_mask, center_node_mask)]]
                .cpu()
                .numpy()
                .tolist()
            )
            center_node_pred_lst.extend(
                softmax_layer(
                    outputs[
                        [v and c for v, c in zip(video_node_mask, center_node_mask)]
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

            last_node_label_lst.extend(
                targets_g[[v and c for v, c in zip(video_node_mask, last_node_mask)]]
                .cpu()
                .numpy()
                .tolist()
            )
            last_node_pred_lst.extend(
                softmax_layer(
                    outputs[[v and c for v, c in zip(video_node_mask, last_node_mask)]]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

        # 统计
        running_loss_g += loss_graph.item()
        running_loss_a += aux_loss_a.item()
        running_loss_v += aux_loss_v.item()
        running_loss_vf += (
            aux_loss_vf.item() if vf_a_emb is not None and vf_v_emb is not None else 0
        )
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_loss_vf = running_loss_vf / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_ap_center_node = average_precision_score(
        center_node_label_lst, center_node_pred_lst
    )
    epoch_ap_last_node = average_precision_score(
        last_node_label_lst, last_node_pred_lst
    )
    print(
        "Train Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, Vf Loss: {:.4f}, mAP: {:.4f}, CNode mAP: {:.4f}, LNode mAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_loss_vf,
            epoch_ap,
            epoch_ap_center_node,
            epoch_ap_last_node,
        )
    )
    return (
        epoch_loss_g,
        epoch_loss_a,
        epoch_loss_v,
        epoch_loss_vf,
        epoch_ap,
        epoch_ap_center_node,
        epoch_ap_last_node,
    )


def _test_model_graph_losses(
    model,
    dataloader,
    criterion: nn.modules.loss._Loss,
    vf_critierion: nn.modules.loss._Loss,
    device,
    ctx_size,
    time_len,  # 图的时间步数
):
    """测试模型，返回图的损失和音频视频的辅助损失"""
    model.eval()
    softmax_layer = torch.nn.Softmax(dim=1)

    pred_lst = []
    label_lst = []
    center_node_pred_lst = []
    center_node_label_lst = []
    last_node_pred_lst = []
    last_node_label_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0
    running_loss_vf = 0.0

    audio_size, _ = dataloader.dataset.get_audio_size()

    for idx, dl in enumerate(dataloader):
        print(
            "\t Val iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_g / (idx + 1)
            ),
            end="\r",
        )

        graph_data = dl
        graph_data = graph_data.to(device)

        center_node_mask = []
        last_node_mask = []
        for c_mask, l_mask in zip(
            graph_data.center_node_mask, graph_data.last_node_mask
        ):
            center_node_mask += c_mask
            last_node_mask += l_mask
        audio_node_mask = []
        for mask in graph_data.audio_node_mask:
            audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        targets_a = graph_data.y[:, 0]
        targets_v = graph_data.y[:, 1]
        targets_g = targets_v.clone()
        targets_g[audio_node_mask] = targets_a[audio_node_mask]
        entities_a = graph_data.y[:, 2]
        entities_v = graph_data.y[:, 3]

        with torch.set_grad_enabled(False):
            outputs, audio_out, video_out, vf_a_emb, vf_v_emb = model(
                graph_data, audio_size
            )
            loss_graph = criterion(outputs, targets_g)
            aux_loss_a = criterion(audio_out, targets_a[audio_node_mask])
            aux_loss_v = criterion(video_out, targets_v[video_node_mask])
            if vf_a_emb is not None and vf_v_emb is not None:
                aux_loss_vf: torch.Tensor = vf_critierion(
                    torch.cat([vf_a_emb, vf_v_emb], dim=0),
                    torch.cat([entities_a, entities_v], dim=0),
                )

            label_lst.extend(targets_g[video_node_mask].cpu().numpy().tolist())
            pred_lst.extend(
                softmax_layer(outputs[video_node_mask]).cpu().numpy()[:, 1].tolist()
            )

            center_node_label_lst.extend(
                targets_g[[v and c for v, c in zip(video_node_mask, center_node_mask)]]
                .cpu()
                .numpy()
                .tolist()
            )
            center_node_pred_lst.extend(
                softmax_layer(
                    outputs[
                        [v and c for v, c in zip(video_node_mask, center_node_mask)]
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

            last_node_label_lst.extend(
                targets_g[[v and c for v, c in zip(video_node_mask, last_node_mask)]]
                .cpu()
                .numpy()
                .tolist()
            )
            last_node_pred_lst.extend(
                softmax_layer(
                    outputs[[v and c for v, c in zip(video_node_mask, last_node_mask)]]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

        # 统计
        running_loss_g += loss_graph.item()
        running_loss_a += aux_loss_a.item()
        running_loss_v += aux_loss_v.item()
        running_loss_vf += (
            aux_loss_vf.item() if vf_a_emb is not None and vf_v_emb is not None else 0
        )

        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_loss_vf = running_loss_vf / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_ap_center_node = average_precision_score(
        center_node_label_lst, center_node_pred_lst
    )
    epoch_ap_last_node = average_precision_score(
        last_node_label_lst, last_node_pred_lst
    )
    print(
        "Val Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, Vf Loss: {:.4f}, mAP: {:.4f}, CNode mAP: {:.4f}, LNode mAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_loss_vf,
            epoch_ap,
            epoch_ap_center_node,
            epoch_ap_last_node,
        )
    )
    return (
        epoch_loss_g,
        epoch_loss_a,
        epoch_loss_v,
        epoch_loss_vf,
        epoch_ap,
        epoch_ap_center_node,
        epoch_ap_last_node,
    )
