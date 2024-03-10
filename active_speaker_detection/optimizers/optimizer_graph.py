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

        (
            train_loss,
            ta_loss,
            tv_loss,
            train_ap,
            train_ap_center_node,
            train_ap_last_node,
        ) = outs_train
        val_loss, va_loss, vv_loss, val_ap, val_ap_center_node, val_ap_last_node = (
            outs_val
        )

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
                    train_ap,
                    train_ap_center_node,
                    train_ap_last_node,
                    val_loss,
                    va_loss,
                    vv_loss,
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
    device,
    ctx_size,
    time_len: int,  # 图的时间步数
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
        targets = graph_data.y[:, 0]

        distance_to_last_graph_list = []
        center_node_mask = []
        last_node_mask = []
        for d_list, c_mask, l_mask in zip(
            graph_data.distance_to_last_graph_list,
            graph_data.center_node_mask,
            graph_data.last_node_mask,
        ):
            distance_to_last_graph_list += d_list
            center_node_mask += c_mask
            last_node_mask += l_mask
        distance_to_last_graph = torch.tensor(
            distance_to_last_graph_list, dtype=torch.float16
        ).to(device)
        max_distance = int(distance_to_last_graph.max().item())

        audio_node_mask = []
        for mask in graph_data.audio_node_mask:
            audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            with autocast(True):
                outputs, *_ = model(graph_data)

                # 分小图计算损失
                loss = torch.zeros(1, device=device)
                total_weight = 0
                for d in range(max_distance + 1):
                    curr_weight = 1 / (d + 1)
                    total_weight += curr_weight
                    # 图的损失
                    loss += curr_weight * criterion(
                        outputs[distance_to_last_graph == d],
                        targets[distance_to_last_graph == d],
                    )
                loss /= total_weight

            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()

        with torch.set_grad_enabled(False):
            label_lst.extend(
                targets[video_node_mask][distance_to_last_graph[video_node_mask] == 0]
                .cpu()
                .numpy()
                .tolist()
            )
            pred_lst.extend(
                softmax_layer(
                    outputs[video_node_mask][
                        distance_to_last_graph[video_node_mask] == 0
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

            center_video_node_mask = [
                v and c for v, c in zip(video_node_mask, center_node_mask)
            ]
            center_node_label_lst.extend(
                targets[center_video_node_mask][
                    distance_to_last_graph[center_video_node_mask] == 0
                ]
                .cpu()
                .numpy()
                .tolist()
            )
            center_node_pred_lst.extend(
                softmax_layer(
                    outputs[center_video_node_mask][
                        distance_to_last_graph[center_video_node_mask] == 0
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

            last_video_node_mask = [
                v and c for v, c in zip(video_node_mask, last_node_mask)
            ]
            last_node_label_lst.extend(
                targets[last_video_node_mask][
                    distance_to_last_graph[last_video_node_mask] == 0
                ]
                .cpu()
                .numpy()
                .tolist()
            )
            last_node_pred_lst.extend(
                softmax_layer(
                    outputs[last_video_node_mask][
                        distance_to_last_graph[last_video_node_mask] == 0
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

        # 统计
        running_loss_g += loss.item()
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_ap_center_node = average_precision_score(
        center_node_label_lst, center_node_pred_lst
    )
    epoch_ap_last_node = average_precision_score(
        last_node_label_lst, last_node_pred_lst
    )
    print(
        "Train Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, mAP: {:.4f}, CNode mAP: {:.4f}, LNode mAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_ap,
            epoch_ap_center_node,
            epoch_ap_last_node,
        )
    )
    return (
        epoch_loss_g,
        epoch_loss_a,
        epoch_loss_v,
        epoch_ap,
        epoch_ap_center_node,
        epoch_ap_last_node,
    )


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
    center_node_pred_lst = []
    center_node_label_lst = []
    last_node_pred_lst = []
    last_node_label_lst = []

    running_loss_g = 0.0
    running_loss_a = 0.0
    running_loss_v = 0.0

    for idx, dl in enumerate(dataloader):
        print(
            "\t Val iter {:d}/{:d} {:.4f}".format(
                idx, len(dataloader), running_loss_g / (idx + 1)
            ),
            end="\r",
        )

        graph_data = dl
        graph_data = graph_data.to(device)
        targets = graph_data.y[:, 0]

        distance_to_last_graph_list = []
        center_node_mask = []
        last_node_mask = []
        for d_list, c_mask, l_mask in zip(
            graph_data.distance_to_last_graph_list,
            graph_data.center_node_mask,
            graph_data.last_node_mask,
        ):
            distance_to_last_graph_list += d_list
            center_node_mask += c_mask
            last_node_mask += l_mask
        distance_to_last_graph = torch.tensor(
            distance_to_last_graph_list, dtype=torch.float16
        ).to(device)
        max_distance = int(distance_to_last_graph.max().item())

        audio_node_mask = []
        for mask in graph_data.audio_node_mask:
            audio_node_mask += mask
        video_node_mask = [not mask for mask in audio_node_mask]

        with torch.set_grad_enabled(False):
            outputs, *_ = model(graph_data)

            # 分小图计算损失
            loss = torch.zeros(1, device=device)
            total_weight = 0
            for d in range(max_distance + 1):
                curr_weight = 1 / (d + 1)
                total_weight += curr_weight
                # 图的损失
                loss += curr_weight * criterion(
                    outputs[distance_to_last_graph == d],
                    targets[distance_to_last_graph == d],
                )
            loss /= total_weight

            label_lst.extend(
                targets[video_node_mask][distance_to_last_graph[video_node_mask] == 0]
                .cpu()
                .numpy()
                .tolist()
            )
            pred_lst.extend(
                softmax_layer(
                    outputs[video_node_mask][
                        distance_to_last_graph[video_node_mask] == 0
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

            center_video_node_mask = [
                v and c for v, c in zip(video_node_mask, center_node_mask)
            ]
            center_node_label_lst.extend(
                targets[center_video_node_mask][
                    distance_to_last_graph[center_video_node_mask] == 0
                ]
                .cpu()
                .numpy()
                .tolist()
            )
            center_node_pred_lst.extend(
                softmax_layer(
                    outputs[center_video_node_mask][
                        distance_to_last_graph[center_video_node_mask] == 0
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

            last_video_node_mask = [
                v and c for v, c in zip(video_node_mask, last_node_mask)
            ]
            last_node_label_lst.extend(
                targets[last_video_node_mask][
                    distance_to_last_graph[last_video_node_mask] == 0
                ]
                .cpu()
                .numpy()
                .tolist()
            )
            last_node_pred_lst.extend(
                softmax_layer(
                    outputs[last_video_node_mask][
                        distance_to_last_graph[last_video_node_mask] == 0
                    ]
                )
                .cpu()
                .numpy()[:, 1]
                .tolist()
            )

        # 统计
        running_loss_g += loss.item()
        if idx == len(dataloader) - 2:
            break

    epoch_loss_g = running_loss_g / len(dataloader)
    epoch_loss_a = running_loss_a / len(dataloader)
    epoch_loss_v = running_loss_v / len(dataloader)
    epoch_ap = average_precision_score(label_lst, pred_lst)
    epoch_ap_center_node = average_precision_score(
        center_node_label_lst, center_node_pred_lst
    )
    epoch_ap_last_node = average_precision_score(
        last_node_label_lst, last_node_pred_lst
    )
    print(
        "Val Graph Loss: {:.4f}, Audio Loss: {:.4f}, Video Loss: {:.4f}, mAP: {:.4f}, CNode mAP: {:.4f}, LNode mAP: {:.4f}".format(
            epoch_loss_g,
            epoch_loss_a,
            epoch_loss_v,
            epoch_ap,
            epoch_ap_center_node,
            epoch_ap_last_node,
        )
    )
    return (
        epoch_loss_g,
        epoch_loss_a,
        epoch_loss_v,
        epoch_ap,
        epoch_ap_center_node,
        epoch_ap_last_node,
    )
