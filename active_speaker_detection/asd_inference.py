#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-22 09:41:42
"""

from typing import Optional

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torchvision import transforms

from active_speaker_detection.models.model import get_backbone

from . import asd_config as asd_conf
from .utils.audio_processing import (
    generate_fbank,
    generate_mel_spectrogram,
    normalize_fbank,
)
from .utils.custom_transforms import video_val


class ActiveSpeakerDetector:

    def __init__(self, trained_model: Optional[str] = None, cpu: bool = False):
        infer_config = asd_conf.inference_params
        self.frames_per_clip = infer_config["frmc"]
        self.ctx_size = infer_config["ctx"]
        self.n_clips = infer_config["nclp"]
        self.strd = infer_config["strd"]
        self.img_size = infer_config["size"]
        self.audio_sample_rate = infer_config["audio_sample_rate"]
        self.is_edge_double = infer_config["is_edge_double"]
        self.is_edge_across_entity = infer_config["is_edge_across_entity"]
        self.device = torch.device("cpu" if cpu else "cuda")

        torch.set_grad_enabled(False)
        self.model, self.encoder, self.encoder_vf = get_backbone(
            infer_config["encoder_type"],
            infer_config["graph_type"],
            infer_config["encoder_enable_vf"],
            infer_config["graph_enable_spatial"],
            train_weights=trained_model,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.video_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                video_val,
            ]
        )

    def detect_active_speaker(
        self,
        faces: list[list[Optional[list[np.ndarray]]]],
        audios: list[np.ndarray],
        face_bboxes: list[list[Optional[tuple[float, float, float, float]]]],
    ) -> list[list[float]]:
        """
        :param faces: 人脸图像，第一个列表是每个时刻的列表，第二个列表是每个人的列表，第三个列表是一个人一个时刻的 clip list，ndarray 形状为 (H, W, C)，BGR
        :param audios: 音频，列表是每个时刻的列表，ndarray 形状为 (N,)，float32 pcm
        :param face_bboxes: 人脸框，第一个列表是每个时刻的列表，第二个列表是每个人的列表，第三个列表是一个人一个时刻的 bbox list，(x1, y1, x2, y2)，相对于原图的比例
        :return: 第一个列表是每个时刻的列表，第二个列表是每个人的列表，每个人每个时刻的 p
        """
        if len(faces) != self.n_clips or len(audios) != self.n_clips:
            raise ValueError("Face and audio data should be of length n_clips")
        if len(faces[0]) != self.ctx_size:
            raise ValueError("Face data should be of length ctx_size")

        faces_tensors = [
            [
                ([self._parse_face(f) for f in clips] if clips is not None else None)
                for clips in faces_per_entity
            ]
            for faces_per_entity in faces
        ]
        # 把一个人的多个画面合并成一个 4D tensor，第一个维度是通道数，第二个维度是clip数，第三四个维度是画面的长宽
        faces_tensors = [
            [
                (torch.stack(fc, dim=1) if fc is not None else None)
                for fc in faces_per_entity
            ]
            for faces_per_entity in faces_tensors
        ]
        # 获取一个不为空的人脸 tensor，用于获取长宽
        example_face_tensor = None
        for i in range(self.ctx_size):
            for j in range(self.n_clips):
                if faces_tensors[j][i] is not None:
                    example_face_tensor = faces_tensors[j][i]
                    break
        assert example_face_tensor is not None

        mel_feats, fbank_feats = zip(*[self._parse_audio(a) for a in audios])

        # 创建一个 tensor，用于存放特征数据，这里是 6D 的，第一个维度是最大节点数，第二个维度分为音视频特征，第三个维度是通道数，第四个维度是clip数，第五六个维度是画面的长宽
        feature_set = torch.zeros(
            self.ctx_size * self.n_clips,
            2,
            example_face_tensor.size(0),
            example_face_tensor.size(1),
            example_face_tensor.size(2),
            example_face_tensor.size(3),
        )

        # 图节点的实体数据
        entity_list: list[int] = []
        # 时间戳列表
        timestamp_list: list[int] = []
        # 位置列表
        position_list: list[tuple[float, float, float, float]] = []

        node_count = 0
        for i in range(self.n_clips - 1, -1, -1):
            audio_data = torch.from_numpy(mel_feats[i])
            for j , face_tensor in enumerate(faces_tensors[i]):
                if face_tensor is None:
                    continue
                feature_set[node_count, 0,0,0, : audio_data.size(1), : audio_data.size(2)] = audio_data
                feature_set[node_count, 1] = face_tensor
                node_count += 1

                entity_list.insert(0, j)
                timestamp_list.insert(0, i)
                position_list.insert(0, face_bboxes[i][j])  # type: ignore

        # 边的出发点，每一条无向边会正反记录两次
        source_vertices: list[int] = []
        # 边的结束点，每一条无向边会正反记录两次
        target_vertices: list[int] = []
        # 边出发点的位置信息，x1, y1, x2, y2
        source_vertices_pos: list[tuple[float, float, float, float]] = []
        # 边结束点的位置信息，x1, y1, x2, y2
        target_vertices_pos: list[tuple[float, float, float, float]] = []

        # 构造边
        for i, (entity, timestamp) in enumerate(zip(entity_list, timestamp_list)):
            for j, (entity, timestamp) in enumerate(zip(entity_list, timestamp_list)):
                # 自己不连接自己
                if i == j:
                    continue

                # 超过了时间步数，不连接
                if abs(i - j) > self.ctx_size:
                    continue

                # 只单向连接
                if not self.is_edge_double and i > j:
                    continue

                if timestamp_list[i] == timestamp_list[j]:
                    # 同一时刻上下文中的实体之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                elif entity_list[i] == entity_list[j]:
                    # 同一实体在不同时刻之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                elif self.is_edge_across_entity:
                    # 不同实体在不同时刻之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])

        graph_data = Data(
            # 维度为 [节点数量, 2, 通道数 , clip数 , 高度 , 宽度]，表示每个节点的音频和视频特征
            x=feature_set,
            # 维度为 [2, 边的数量]，表示每条边的两侧节点的索引
            edge_index=torch.tensor(
                [source_vertices, target_vertices], dtype=torch.long
            ),
            # 维度为 [边的数量, 2, 4]，表示每条边的两侧节点的位置信息
            edge_attr=torch.tensor(
                [source_vertices_pos, target_vertices_pos], dtype=torch.float
            ).transpose(0, 1),
        )
        graph_output, *_ = self.model(
            graph_data, self.ctx_size, mel_feats[0].shape, fbank_feats[0].shape
        )
        graph_output = F.softmax(graph_output, dim=1)

        output_list = [[0.0 for _ in faces_per_entity] for faces_per_entity in faces]
        for i in range(graph_output.size(0)):
            output_list[timestamp_list[i]][entity_list[i]] = graph_output[i, 1].item()

        return output_list

    def _parse_face(self, face: np.ndarray):
        # BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = transforms.ToPILImage()(face)
        return self.video_transform(face_pil)

    def _parse_audio(self, audio: np.ndarray):
        # float32 pcm 转 int16
        audio = (audio * 32768).astype(np.int16)
        mel_feat = generate_mel_spectrogram(audio, self.audio_sample_rate)
        fbank_feat = generate_fbank(audio, self.audio_sample_rate)
        fbank_feat = normalize_fbank(fbank_feat, torch.FloatTensor([1.0]))
        return np.float32(mel_feat), np.float32(fbank_feat)
