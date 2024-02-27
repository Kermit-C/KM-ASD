#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import csv
import os
import pickle
from typing import Dict, List, Tuple


class DataStoreLoader:
    """数据存储加载器"""

    """
    csv 标注格式：
    | video_id | frame_timestamp | entity_box_x1 | entity_box_y1 | entity_box_x2 | entity_box_y2 | label | entity_id | label_id | instance_id |

    标签：
    0: NOT_SPEAKING
    1: SPEAKING_AUDIBLE
    2: SPEAKING_NOT_AUDIBLE
    """
    def __init__(self, audio_root, video_root, csv_file_path):
        # 数据根目录
        self.audio_root = audio_root  # 音频根目录
        self.video_root = video_root  # 视频根目录

        # 一个实体对应 video+entity

        # 一个人在什么时候是否说话 (entity_id, timestamp, entity_label)
        self.entity_data: Dict[str, Dict[str, List[Tuple[str, str, int]]]] = {}
        # 一个人在什么时候的位置信息 (x1, y1, x2, y2)
        self.entity_pos_data: Dict[
            str, Dict[str, List[Tuple[float, float, float, float]]]
        ] = {}
        # 一个时间点是否说话，用于音频节点
        self.speech_data: Dict[str, Dict[str, int]] = {}
        # 一个时间点的实体们
        self.ts_to_entity: Dict[str, Dict[str, List[str]]] = {}

        # 所有实体的列表 (video_id, entity_id)
        self.entity_list: List[Tuple[str, str]] = []
        # 所有特征的列表 (video_id, entity_id, timestamp, entity_label)
        self.feature_list: List[Tuple[str, str, str, int]] = []

        # 读取并处理标签文件到 cache
        entity_set = self.cache_entity_data(csv_file_path)
        self.postprocess_entity_list(entity_set)

    def cache_entity_data(self, csv_file_path):
        """读取标签文件到 cache，返回所有实体的 set"""
        # 保存所有实体的 set, (video_id, entity_id)
        entity_set: set[Tuple[str, str]] = set()

        csv_data = self.csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        # csv_data 是一个二维列表，每一行是一个列表，每一行的元素是一个字符串
        for csv_row in csv_data:
            if len(csv_row) != 10:
                # 如果不是 10 个元素，就跳过
                continue
            video_id = csv_row[0]  # 视频 id
            entity_id = csv_row[-3]  # 实体 id
            timestamp = csv_row[1]  # 时间戳

            speech_label = self.postprocess_speech_label(csv_row[-2])
            entity_label = self.postprocess_entity_label(csv_row[-2])
            # 最小的实体数据，什么人在什么时间什么状态
            minimal_entity_data = (
                entity_id,
                timestamp,
                entity_label,
            )
            # 最小的实体位置数据，什么人在什么时间什么位置
            minimal_entity_pos_data = (
                float(csv_row[2]),
                float(csv_row[3]),
                float(csv_row[4]),
                float(csv_row[5]),
            )

            # 存储最少的实体数据
            # 先判断 video_id 是否在 entity_data 中，如果不在，就添加一个空字典
            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}
                self.entity_pos_data[video_id] = {}
            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
                self.entity_pos_data[video_id][entity_id] = []
                entity_set.add((video_id, entity_id))
            # 将 minimal_entity_data 添加到 entity_data 中
            self.entity_data[video_id][entity_id].append(minimal_entity_data)
            self.entity_pos_data[video_id][entity_id].append(minimal_entity_pos_data)

            # 存储语音元数据
            if video_id not in self.speech_data.keys():
                self.speech_data[video_id] = {}
            if timestamp not in self.speech_data[video_id].keys():
                self.speech_data[video_id][timestamp] = speech_label
            # 相当于聚合一个时间点所有人说话状态
            new_speech_label = max(self.speech_data[video_id][timestamp], speech_label)
            self.speech_data[video_id][timestamp] = new_speech_label

        return entity_set

    def postprocess_entity_list(self, entity_set):
        """后处理 _cache_entity_data 得到的数据"""
        print("Initial", len(entity_set))

        # 过滤掉磁盘上没有的实体，video_root 中有每一个实体的文件夹
        print("video_root", self.video_root)
        all_disk_data = set(os.listdir(self.video_root))  # type: ignore
        for video_id, entity_id in entity_set.copy():
            if entity_id.replace(":", "_") not in all_disk_data:
                entity_set.remove((video_id, entity_id))
        print("Pruned not in disk", len(entity_set))

        # 过滤掉实体文件夹中，timestamp 画面数和 entity_data 中的数量不一致的实体
        for video_id, entity_id in entity_set.copy():
            dir = os.path.join(self.video_root, entity_id.replace(":", "_"))  # type: ignore
            if len(os.listdir(dir)) != len(self.entity_data[video_id][entity_id]):
                entity_set.remove((video_id, entity_id))

        print("Pruned not complete", len(entity_set))
        self.entity_list = sorted(list(entity_set))

        for video_id, entity_id in entity_set:
            ent_min_data = self.entity_data[video_id][entity_id]

            # 保存时间点对应实体列表的关系字典
            if video_id not in self.ts_to_entity.keys():
                self.ts_to_entity[video_id] = {}
            for ed in ent_min_data:
                timestamp = ed[1]
                if timestamp not in self.ts_to_entity[video_id].keys():
                    self.ts_to_entity[video_id][timestamp] = []
                self.ts_to_entity[video_id][timestamp].append(entity_id)

            # 存储所有特征的列表
            for ed in ent_min_data:
                self.feature_list.append((video_id, entity_id, ed[1], ed[2]))

    def postprocess_speech_label(self, speech_label):
        speech_label = int(speech_label)
        if speech_label == 2:
            # 把说话但没有声音也标记为未说话
            speech_label = 0
        return speech_label

    def postprocess_entity_label(self, entity_label):
        entity_label = int(entity_label)
        if entity_label == 2:
            # 把说话但没有声音也标记为未说话
            entity_label = 0
        return entity_label

    def csv_to_list(self, csv_path: str) -> List[List[str]]:
        """读取 csv 文件，返回 list 格式的数据"""
        as_list = None
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            as_list = list(reader)
        return as_list


if __name__ == "__main__":
    out_path = "active_speaker_detection/datasets/resources/data_store_cache"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 训练集

    audio_root = "/ssd1/ckm2/instance_wavs_time_train/"
    video_root = "/ssd1/ckm2/instance_crops_time_train/"
    csv_file_path = "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_train_augmented.csv"
    datastore = DataStoreLoader(
        audio_root=audio_root, video_root=video_root, csv_file_path=csv_file_path
    )
    dataset_store_cache = {
        "entity_data": datastore.entity_data,
        "entity_pos_data": datastore.entity_pos_data,
        "speech_data": datastore.speech_data,
        "ts_to_entity": datastore.ts_to_entity,
        "entity_list": datastore.entity_list,
        "feature_list": datastore.feature_list,
    }
    with open(os.path.join(out_path, "dataset_train_store_cache.pkl"), "wb") as f:
        pickle.dump(dataset_store_cache, f)

    print(
        "训练集数据存储加载器已经保存到: ",
        os.path.join(out_path, "dataset_train_store_cache.pkl"),
    )

    # 验证集

    audio_root = "/ssd1/ckm2/instance_wavs_time_val/"
    video_root = "/ssd1/ckm2/instance_crops_time_val/"
    csv_file_path = (
        "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_val_augmented.csv"
    )
    datastore = DataStoreLoader(
        audio_root=audio_root, video_root=video_root, csv_file_path=csv_file_path
    )
    dataset_store_cache = {
        "entity_data": datastore.entity_data,
        "entity_pos_data": datastore.entity_pos_data,
        "speech_data": datastore.speech_data,
        "ts_to_entity": datastore.ts_to_entity,
        "entity_list": datastore.entity_list,
        "feature_list": datastore.feature_list,
    }
    with open(os.path.join(out_path, "dataset_val_store_cache.pkl"), "wb") as f:
        pickle.dump(dataset_store_cache, f)

    print(
        "验证集数据存储加载器已经保存到: ",
        os.path.join(out_path, "dataset_val_store_cache.pkl"),
    )

    print("完成！")
