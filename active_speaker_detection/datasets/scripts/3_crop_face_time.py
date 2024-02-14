#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-08 09:44:55
"""

import glob
import os
import random
from typing import List

import cv2
import numpy as np
import pandas as pd


def generate_mini_dataset(video_dir, output_dir, df, balanced=False):
    # 假设负样本总是多于正样本
    df_neg = df[df["label_id"] == 0]
    df_pos = df[df["label_id"] == 1]
    instances_neg = df_neg["instance_id"].unique().tolist()
    instances_pos = df_pos["instance_id"].unique().tolist()

    if balanced:
        random.seed(17)
        instances_neg = random.sample(instances_neg, len(instances_pos))
        df_neg = df_neg[df["instance_id"].isin(instances_neg)]

    print(len(instances_pos), len(instances_neg))
    print(len(df_pos), len(df_neg))
    balanced_df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    balanced_df = balanced_df.sort_values(["entity_id", "frame_timestamp"]).reset_index(
        drop=True
    )
    entity_list: List[str] = balanced_df["entity_id"].unique().tolist()
    # 聚合每个实体的数据
    balanced_gb = balanced_df.groupby("entity_id")

    # 创建文件夹
    for l in balanced_df["label"].unique().tolist():
        d = os.path.join(output_dir, l)
        if not os.path.isdir(d):
            os.makedirs(d)

    for entity_idx, instance in enumerate(entity_list):
        instance_data = balanced_gb.get_group(instance)

        video_key = instance_data.iloc[0]["video_id"]
        entity_id = instance_data.iloc[0]["entity_id"]
        video_file = glob.glob(os.path.join(video_dir, "{}.*".format(video_key)))[0]

        V = cv2.VideoCapture(video_file)

        # 创建实体文件夹
        instance_dir = os.path.join(
            os.path.join(output_dir, entity_id.replace(":", "_"))
        )
        if not os.path.isdir(instance_dir):
            os.makedirs(instance_dir)

        j = 0
        for _, row in instance_data.iterrows():
            image_filename = os.path.join(
                instance_dir, str(row["frame_timestamp"]) + ".jpg"
            )
            if os.path.exists(image_filename):
                print("skip", image_filename)
                continue

            V.set(cv2.CAP_PROP_POS_MSEC, row["frame_timestamp"] * 1e3)

            # 读取视频帧
            _, frame = V.read()
            h = np.size(frame, 0)
            w = np.size(frame, 1)

            # 裁剪人脸
            crop_x1 = int(row["entity_box_x1"] * w)
            crop_y1 = int(row["entity_box_y1"] * h)
            crop_x2 = int(row["entity_box_x2"] * w)
            crop_y2 = int(row["entity_box_y2"] * h)
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]

            j = j + 1

            cv2.imwrite(image_filename, face_crop)

        print(f"视频切片进度：{entity_idx + 1}/{len(entity_list)}", end="\r")


if __name__ == "__main__":
    ava_video_dir = "active_speaker_detection/datasets/resources/videos"
    video_list = [
        {
            "output_dir": "active_speaker_detection/datasets/resources/crops/instance_crops_time_train",
            "csv": "active_speaker_detection/datasets/resources/annotations/ava_activespeaker_train_augmented.csv",
        },
        {
            "output_dir": "active_speaker_detection/datasets/resources/crops/instance_crops_time_val",
            "csv": "active_speaker_detection/datasets/resources/annotations/ava_activespeaker_val_augmented.csv",
        },
        {
            "output_dir": "active_speaker_detection/datasets/resources/crops/instance_crops_time_test",
            "csv": "active_speaker_detection/datasets/resources/annotations/ava_activespeaker_test_augmented.csv",
        },
    ]

    for video in video_list:
        df = pd.read_csv(video["csv"], engine="python")
        train_subset_dir = os.path.join(video["output_dir"])
        generate_mini_dataset(ava_video_dir, train_subset_dir, df, balanced=False)
        print(f"完成了{video['csv']}的视频切片")

    print("完成！")
