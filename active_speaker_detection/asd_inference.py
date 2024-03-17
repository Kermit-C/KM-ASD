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
        self.max_context = infer_config["ctx"]
        self.graph_time_steps = infer_config["nclp"]
        self.graph_time_stride = infer_config["strd"]
        self.img_size = infer_config["size"]
        self.audio_sample_rate = infer_config["audio_sample_rate"]
        self.is_edge_double = infer_config["is_edge_double"]
        self.is_edge_across_entity = infer_config["is_edge_across_entity"]
        self.encoder_enable_vf = infer_config["encoder_enable_vf"]
        self.device = torch.device("cpu" if cpu else "cuda")

        torch.set_grad_enabled(False)
        self.model, self.encoder, self.encoder_vf = get_backbone(
            infer_config["encoder_type"],
            infer_config["graph_type"],
            self.encoder_enable_vf,
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

    def gen_feature(
        self, faces: list[list[np.ndarray]], audios: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        :param faces: 人脸图像，第一个列表是每个人的列表，第二个列表是一个人一个时刻的 clip list，ndarray 形状为 (H, W, C)，BGR
        :param audios: 音频，ndarray 形状为 (N,)，float32 pcm
        """
        faces_tensors = [[self._parse_face(f) for f in clips] for clips in faces]
        # 把一个人的多个画面合并成一个 4D tensor，第一个维度是通道数，第二个维度是clip数，第三四个维度是画面的长宽
        faces_tensor = torch.stack([torch.stack(fc, dim=1) for fc in faces_tensors]).to(
            self.device
        )

        mel_feats, _ = zip(*[self._parse_audio(a) for a in audios])
        audios_tensor = torch.from_numpy(np.stack(mel_feats, axis=0)).to(self.device)

        audio_feats, video_feats = self.model.forward_encoder(
            faces_tensor, audios_tensor
        )

        if self.encoder_enable_vf:
            vf_a_emb, vf_v_emb = self.model.forward_encoder_vf(audio_feats, video_feats)
            return (
                [i for i in audio_feats.cpu().detach().numpy()],
                [i for i in video_feats.cpu().detach().numpy()],
                [i for i in vf_a_emb.cpu().detach().numpy()],
                [i for i in vf_v_emb.cpu().detach().numpy()],
            )
        else:
            return (
                [i for i in audio_feats.cpu().detach().numpy()],
                [i for i in video_feats.cpu().detach().numpy()],
                [],
                [],
            )

    def detect_active_speaker(
        self,
        faces: list[list[list[np.ndarray]]],
        audios: list[np.ndarray],
        face_bboxes: list[list[tuple[float, float, float, float]]],
        faces_vf_emb: Optional[list[list[list[np.ndarray]]]] = None,
        audios_vf_emb: Optional[list[np.ndarray]] = None,
    ) -> list[list[float]]:
        """
        :param faces: 人脸图像，第一个列表是每个时刻的列表，第二个列表是每个人的列表，第三个列表是一个人一个时刻的 feat
        :param audios: 音频，列表是每个时刻的 feat，ndarray 形状为 (N,)
        :param face_bboxes: 人脸框，第一个列表是每个时刻的列表，第二个列表是每个人的列表，第三个列表是一个人一个时刻的 bbox list，(x1, y1, x2, y2)，相对于原图的比例
        :return: 第一个列表是每个时刻的列表，第二个列表是每个人的列表，每个人每个时刻的 p
        """
        if len(faces) != self.graph_time_steps or len(audios) != self.graph_time_steps:
            raise ValueError("Face and audio data should be of length graph_time_steps")
        if len(faces[0]) != self.max_context:
            raise ValueError("Face data should be of length max_context")

        max_dim = 0
        max_dim = max(faces[0][0][0].shape[0], max_dim)
        max_dim = max(audios[0].shape[0], max_dim)
        if self.encoder_enable_vf:
            assert faces_vf_emb is not None and audios_vf_emb is not None
            max_dim = max(faces_vf_emb[0][0][0].shape[0], max_dim)
            max_dim = max(audios_vf_emb[0].shape[0], max_dim)

        # 创建一个 tensor，用于存放特征数据，这里是 6D 的，第一个维度是最大节点数，第二个维度分为音视频特征，第三个维度是维度数
        feature_set = torch.zeros(
            (self.max_context + 1) * self.graph_time_steps,
            4 if self.encoder_enable_vf else 2,
            max_dim,
        ).to(self.device)

        # 图节点的实体数据
        entity_list: list[int] = []
        # 纯音频图节点掩码
        audio_feature_mask: list[bool] = []
        # 节点的时刻对应纯音频节点的索引
        audio_feature_idx_list: list[int] = []
        # 时间戳列表
        timestamp_list: list[int] = []
        # 位置列表
        position_list: list[tuple[float, float, float, float]] = []

        node_count = 0
        for i in range(self.graph_time_steps - 1, -1, -1):
            audio_data = torch.from_numpy(audios[i])
            audio_feature_idx = node_count
            for j, face_np in enumerate(faces[i]):
                if j == 0:
                    # 第一个节点之前加一个纯音频节点
                    feature_set[node_count, 0] = audio_data
                    if self.encoder_enable_vf:
                        assert audios_vf_emb is not None
                        feature_set[node_count, 2] = torch.from_numpy(audios_vf_emb[i])
                    entity_list.insert(0, -1)
                    audio_feature_mask.insert(0, True)
                    audio_feature_idx_list.insert(0, audio_feature_idx)
                    timestamp_list.insert(0, i)
                    position_list.insert(0, (0, 0, 1, 1))
                    node_count += 1

                feature_set[node_count, 0] = audio_data
                feature_set[node_count, 1] = torch.from_numpy(face_np)
                if self.encoder_enable_vf:
                    assert faces_vf_emb is not None and audios_vf_emb is not None
                    feature_set[node_count, 2] = torch.from_numpy(audios_vf_emb[i])
                    feature_set[node_count, 3] = torch.from_numpy(faces_vf_emb[i][j])

                entity_list.insert(0, j)
                audio_feature_mask.insert(0, False)
                audio_feature_idx_list.insert(0, audio_feature_idx)
                timestamp_list.insert(0, i)
                position_list.insert(0, face_bboxes[i][j])  # type: ignore

                node_count += 1
        feature_set = feature_set[:node_count]

        # 边的出发点，每一条无向边会正反记录两次
        source_vertices: list[int] = []
        # 边的结束点，每一条无向边会正反记录两次
        target_vertices: list[int] = []
        # 边出发点的位置信息，x1, y1, x2, y2
        source_vertices_pos: list[tuple[float, float, float, float]] = []
        # 边结束点的位置信息，x1, y1, x2, y2
        target_vertices_pos: list[tuple[float, float, float, float]] = []
        # 边出发点是否是音频特征
        source_vertices_audio: list[int] = []
        # 边结束点是否是音频特征
        target_vertices_audio: list[int] = []
        # 边的时间差比例
        time_delta_rate: list[float] = []
        # 边的时间差
        time_delta: list[int] = []
        # 是否自己连接边
        self_connect: list[int] = []

        # 构造边
        for i, (entity_i, timestamp_i) in enumerate(zip(entity_list, timestamp_list)):
            for j, (entity_j, timestamp_j) in enumerate(
                zip(entity_list, timestamp_list)
            ):
                if timestamp_i == timestamp_j:
                    # 同一时刻上下文中的实体之间的连接
                    source_vertices.append(i)
                    target_vertices.append(j)
                    source_vertices_pos.append(position_list[i])
                    target_vertices_pos.append(position_list[j])
                    source_vertices_audio.append(1 if audio_feature_mask[i] else 0)
                    target_vertices_audio.append(1 if audio_feature_mask[j] else 0)
                    time_delta_rate.append(
                        abs(timestamp_i - timestamp_j) / self.graph_time_steps
                    )
                    time_delta.append(
                        abs(timestamp_i - timestamp_j) * self.graph_time_stride
                    )
                    self_connect.append(int(i == j))
                else:
                    # 超过了时间步数，不连接
                    if abs(timestamp_i - timestamp_j) > self.graph_time_steps:
                        continue

                    # 只单向连接
                    if not self.is_edge_double and timestamp_i > timestamp_j:
                        continue

                    # if entity_i == entity_j:
                    # 使用间隔判断是否同一实体，避免空白实体交叉连接
                    if abs(i - j) % (self.max_context + 1) == 0:
                        # 同一实体在不同时刻之间的连接
                        source_vertices.append(i)
                        target_vertices.append(j)
                        source_vertices_pos.append(position_list[i])
                        target_vertices_pos.append(position_list[j])
                        source_vertices_audio.append(1 if audio_feature_mask[i] else 0)
                        target_vertices_audio.append(1 if audio_feature_mask[j] else 0)
                        time_delta_rate.append(
                            abs(timestamp_i - timestamp_j) / self.graph_time_steps
                        )
                        time_delta.append(
                            abs(timestamp_i - timestamp_j) * self.graph_time_stride
                        )
                        self_connect.append(int(i == j))
                    elif self.is_edge_across_entity:
                        # 不同实体在不同时刻之间的连接
                        source_vertices.append(i)
                        target_vertices.append(j)
                        source_vertices_pos.append(position_list[i])
                        target_vertices_pos.append(position_list[j])
                        source_vertices_audio.append(1 if audio_feature_mask[i] else 0)
                        target_vertices_audio.append(1 if audio_feature_mask[j] else 0)
                        time_delta_rate.append(
                            abs(timestamp_i - timestamp_j) / self.graph_time_steps
                        )
                        time_delta.append(
                            abs(timestamp_i - timestamp_j) * self.graph_time_stride
                        )
                        self_connect.append(int(i == j))

        graph_data = Data(
            # 维度为 [节点数量, 4 or 2, n]，表示每个节点的音频、视频特征、音频音脸嵌入、视频音脸嵌入
            x=feature_set,
            # 维度为 [2, 边的数量]，表示每条边的两侧节点的索引
            edge_index=torch.tensor(
                [source_vertices, target_vertices], dtype=torch.long, device=self.device
            ),
            # 维度为 [边的数量, 6, 4]，表示每条边的两侧节点的位置信息、两侧节点是否纯音频节点、时间差比例、时间差、是否自己连接
            edge_attr=torch.tensor(
                [
                    source_vertices_pos,
                    target_vertices_pos,
                    [
                        (s_audio, t_audio, 0, 0)
                        for s_audio, t_audio in zip(
                            source_vertices_audio, target_vertices_audio
                        )
                    ],
                    [(rate, 0, 0, 0) for rate in time_delta_rate],
                    [(delta, 0, 0, 0) for delta in time_delta],
                    [(self_c, 0, 0, 0) for self_c in self_connect],
                ],
                dtype=torch.float,
                device=self.device,
            ).transpose(0, 1),
            # 纯音频节点的掩码
            audio_node_mask=audio_feature_mask,
            # 节点的时刻对应纯音频节点的索引
            audio_feature_idx_list=audio_feature_idx_list,
        )
        graph_output, *_ = self.model(graph_data)
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
