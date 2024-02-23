#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: Kermit
@Date: 2024-02-22 09:41:42
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torchvision import transforms

from active_speaker_detection.models.graph_model import get_backbone

from . import asd_config as asd_conf
from .models.graph_layouts import (
    generate_av_mask,
    get_spatial_connection_pattern,
    get_temporal_connection_pattern,
)
from .utils.audio_processing import (
    generate_fbank,
    generate_mel_spectrogram,
    normalize_fbank,
)
from .utils.custom_transforms import video_val


class ActiveSpeakerDetector:

    def __init__(self, trained_model: str, cpu: bool = False):
        infer_config = asd_conf.inference_params
        self.frames_per_clip = infer_config["frmc"]
        self.ctx_size = infer_config["ctx"]
        self.n_clips = infer_config["nclp"]
        self.strd = infer_config["strd"]
        self.img_size = infer_config["size"]
        self.audio_sample_rate = infer_config["audio_sample_rate"]
        self.device = torch.device("cpu" if cpu else "cuda")

        self.spatial_connection_pattern = get_spatial_connection_pattern(
            self.ctx_size, self.n_clips
        )
        self.temporal_connection_pattern = get_temporal_connection_pattern(
            self.ctx_size, self.n_clips
        )
        self.spatial_batch_edges = torch.tensor(
            [
                self.spatial_connection_pattern["src"],
                self.spatial_connection_pattern["dst"],
            ],
            dtype=torch.long,
        )
        self.temporal_batch_edges = torch.tensor(
            [
                self.temporal_connection_pattern["src"],
                self.temporal_connection_pattern["dst"],
            ],
            dtype=torch.long,
        )

        torch.set_grad_enabled(False)
        self.model = get_backbone(
            infer_config["encoder_type"], train_weights=trained_model
        ).to(self.device)
        self.model.eval()

        self.video_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                video_val,
            ]
        )

    def detect_active_speaker(
        self,
        faces: list[list[list[np.ndarray]]],
        audios: list[np.ndarray],
    ) -> list[list[float]]:
        """
        :param faces: 人脸图像，第一个列表是每个时刻的列表，第二个列表是每个人的列表，第三个列表是一个人一个时刻的 clip list，ndarray 形状为 (H, W, C)，BGR
        :param audios: 音频，列表是每个时刻的列表，ndarray 形状为 (N,)，float32 pcm
        :return: 第一个列表是每个时刻的列表，第二个列表是每个人的列表，每个人每个时刻的 p
        """
        if len(faces) != self.n_clips or len(audios) != self.n_clips:
            raise ValueError("Face and audio data should be of length n_clips")
        if len(faces[0]) != self.ctx_size:
            raise ValueError("Face data should be of length ctx_size")

        faces_tensors = [
            [[self._parse_face(f) for f in clips] for clips in faces_per_entity]
            for faces_per_entity in faces
        ]
        # 把一个人的多个画面合并成一个 4D tensor
        faces_tensors = [
            [torch.stack(fc, dim=1) for fc in faces_per_entity]
            for faces_per_entity in faces_tensors
        ]

        mel_feats, fbank_feats = zip(*[self._parse_audio(a) for a in audios])

        nodes_per_time = self.ctx_size + 1
        # 创建一个 tensor，用于存放特征数据，这里是 5D 的，第一个维度是时刻上下文节点数*时间上下文数，第二个维度是通道数，第三个维度是clip数，第四五个维度是画面的长宽
        feature_set = torch.zeros(
            nodes_per_time * self.n_clips,
            faces_tensors[0][0].size(0),
            faces_tensors[0][0].size(1),
            faces_tensors[0][0].size(2),
            faces_tensors[0][0].size(3),
        )

        for i in range(self.n_clips):
            graph_offset = i * nodes_per_time
            # 第一个维度的第一个节点是音频特征
            audio_data = torch.from_numpy(mel_feats[i])
            feature_set[
                graph_offset, 0, 0, : audio_data.size(1), : audio_data.size(2)
            ] = audio_data
            # 填充视频特征
            for j in range(self.ctx_size):
                feature_set[graph_offset + (i + 1), ...] = faces_tensors[i][j]

        graph_data = Data(
            x=feature_set,
            edge_index=(self.spatial_batch_edges, self.temporal_batch_edges),  # type: ignore
        )
        graph_output, *_ = self.model(
            graph_data, self.ctx_size, mel_feats[0].shape, fbank_feats[0].shape
        )
        graph_output = F.softmax(graph_output, dim=1)

        _, face_mask = generate_av_mask(self.ctx_size, self.n_clips)
        return (
            graph_output[face_mask][:, 1].reshape(self.n_clips, self.ctx_size).tolist()
        )

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
