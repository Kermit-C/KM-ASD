#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-08 09:44:55
"""

import os
import random

import pandas as pd
from scipy.io import wavfile

# 从 extract_audio_tracks 中提取出来的就是 16khz 的音频
sampling_rate = 16000


def generate_audio_meta_data(full_df, balanced=True, random_seed=42):
    # 假设负样本总是多于正样本
    df_neg = full_df[full_df["label_id"] == 0]
    df_pos = full_df[full_df["label_id"] == 1]
    instances_neg = df_neg["instance_id"].unique().tolist()
    instances_pos = df_pos["instance_id"].unique().tolist()

    if balanced:
        random.seed(17)
        instances_neg = random.sample(instances_neg, len(instances_pos))
        df_neg = df_neg[full_df["instance_id"].isin(instances_neg)]

    print(len(instances_pos), len(instances_neg))
    print(len(df_pos), len(df_neg))
    balanced_df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    balanced_df = balanced_df.sort_values(["entity_id", "frame_timestamp"]).reset_index(
        drop=True
    )
    entity_list = balanced_df["entity_id"].unique().tolist()
    balanced_gb = balanced_df.groupby("entity_id")
    return balanced_gb, entity_list


def extract_audio_tracks_time(audio_dir, output_dir, balanced_gb, entity_list):
    audio_features = {}

    # 创建文件夹
    for l in balanced_gb["entity_id"].unique().tolist():
        d = os.path.join(output_dir, l[0].replace(":", "_"))
        if not os.path.isdir(d):
            os.makedirs(d)

    for entity_idx, entity in enumerate(entity_list):
        instance_data = balanced_gb.get_group(entity)

        video_key = instance_data.iloc[0]["video_id"]
        start = instance_data.iloc[0]["frame_timestamp"]
        end = instance_data.iloc[-1]["frame_timestamp"]
        entity_id = instance_data.iloc[0]["entity_id"]
        instance_path = os.path.join(output_dir, entity_id.replace(":", "_") + ".wav")

        if video_key not in audio_features.keys():
            print("cache audio for ", video_key)
            audio_file = os.path.join(audio_dir, video_key + ".wav")
            sample_rate, audio = wavfile.read(audio_file)
            print(sample_rate, len(audio))
            audio_features[video_key] = audio

        audio_start = int(float(start) * sampling_rate)
        audio_end = int(float(end) * sampling_rate)

        audio_data = audio_features[video_key][audio_start:audio_end]
        wavfile.write(instance_path, sampling_rate, audio_data)

        print(f"音频切片进度：{entity_idx + 1}/{len(entity_list)}", end="\r")


if __name__ == "__main__":
    ava_audio_dir = "active_speaker_detection/datasets/resources/audio_tracks"
    audio_list = [
        {
            "output_dir": "active_speaker_detection/datasets/resources/crops/instance_wavs_time_train",
            "csv": "active_speaker_detection/datasets/resources/annotations/annotations/ava_activespeaker_train_augmented.csv",
        },
        {
            "output_dir": "active_speaker_detection/datasets/resources/crops/instance_wavs_time_val",
            "csv": "active_speaker_detection/datasets/resources/annotations/annotations/ava_activespeaker_val_augmented.csv",
        },
        {
            "output_dir": "active_speaker_detection/datasets/resources/crops/instance_wavs_time_test",
            "csv": "active_speaker_detection/datasets/resources/annotations/annotations/ava_activespeaker_test_augmented.csv",
        },
    ]

    for audio in audio_list:
        df = pd.read_csv(audio["csv"])
        sorted_df, entity_list = generate_audio_meta_data(df, balanced=False)
        extract_audio_tracks_time(
            ava_audio_dir, audio["output_dir"], sorted_df, entity_list
        )
        print(f"完成了{audio['csv']}的音频切片")

    print("完成！")
