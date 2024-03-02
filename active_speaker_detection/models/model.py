#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter

from active_speaker_detection.models.graph_rnn_net import GraphRnnNet
from active_speaker_detection.models.two_stream_mobilev1_net import get_mobilev1_encoder
from active_speaker_detection.models.two_stream_mobilev2_net import get_mobilev2_encoder
from active_speaker_detection.models.two_stream_resnext_net import get_resnext_encoder

from ..utils.spatial_grayscale_util import batch_create_spatial_grayscale
from .graph_all_edge_net import GraphAllEdgeNet
from .graph_all_edge_weight_net import GraphAllEdgeWeightNet
from .graph_gat_edge_net import GraphGatEdgeNet
from .graph_gated_edge_net import GraphGatedEdgeNet
from .graph_gin_edge_net import GraphGinEdgeNet
from .graph_transformer_edge_net import GraphTransformerEdgeNet
from .spatial_mobilenet_net import get_spatial_mobilenet_net
from .spatial_net import get_spatial_net
from .two_stream_light_net import get_light_encoder
from .two_stream_resnet_net import get_resnet_encoder
from .two_stream_resnet_tsm_net import get_resnet_tsm_encoder


class MyModel(nn.Module):

    def __init__(
        self,
        encoder,
        graph_net,
        encoder_enable_vf: bool,
        graph_enable_spatial: bool,
        spatial_net: Optional[nn.Module] = None,  # 为空则不使用空间关系网络
        spatial_feature_dim: int = 64,
        spatial_grayscale_width: int = 32,
        spatial_grayscale_height: int = 32,
    ):
        super().__init__()

        self.encoder = encoder
        self.graph_net = graph_net

        self.encoder_enable_vf = encoder_enable_vf
        self.graph_enable_spatial = graph_enable_spatial

        self.spatial_net = spatial_net
        self.spatial_bn = nn.BatchNorm1d(spatial_feature_dim)

        self.spatial_feature_dim = spatial_feature_dim
        self.spatial_grayscale_width = spatial_grayscale_width
        self.spatial_grayscale_height = spatial_grayscale_height
        self.spatial_mini_batch = 32  # 空间网络的 mini batch 大小，因为图节点数太大了（上千），所以需要分批次计算

    def forward(
        self,
        data,
        audio_size: Optional[tuple[int, int, int]] = None,
    ):
        x, edge_attr_info = data.x, data.edge_attr

        if len(x.shape) > 3:
            # 从数据中提取音频和视频特征
            assert audio_size is not None
            audio_data = x[:, 0, 0, 0, : audio_size[1], : audio_size[2]].unsqueeze(1)
            video_data = x[:, 1, :, :, :, :]
            audio_feats, video_feats, audio_out, video_out, _, vf_a_emb, vf_v_emb = (
                self.encoder(audio_data, video_data)
            )
            audio_feat_dim = audio_feats.size(1)
            video_feat_dim = video_feats.size(1)
            if self.encoder_enable_vf:
                vf_a_emb_dim = vf_a_emb.size(1)
                vf_v_emb_dim = vf_v_emb.size(1)
                max_dim = max(
                    audio_feat_dim, video_feat_dim, vf_a_emb_dim, vf_v_emb_dim
                )
                # 维度不够的在后面填充 0
                audio_feats = F.pad(audio_feats, (0, max_dim - audio_feat_dim))
                video_feats = F.pad(video_feats, (0, max_dim - video_feat_dim))
                vf_a_emb = F.pad(vf_a_emb, (0, max_dim - vf_a_emb_dim))
                vf_v_emb = F.pad(vf_v_emb, (0, max_dim - vf_v_emb_dim))
                data.x = torch.stack(
                    [audio_feats, video_feats, vf_a_emb, vf_v_emb], dim=1
                )
            else:
                max_dim = max(audio_feat_dim, video_feat_dim)
                audio_feats = F.pad(audio_feats, (0, max_dim - audio_feat_dim))
                video_feats = F.pad(video_feats, (0, max_dim - video_feat_dim))
                data.x = torch.stack([audio_feats, video_feats], dim=1)
        else:
            # 输入的就是 encoder 出来的特征
            audio_out = None
            video_out = None
            vf_a_emb = None
            vf_v_emb = None
            if not self.encoder_enable_vf:
                audio_feats = x[:, 0, :]
                video_feats = x[:, 1, :]
                data.x = torch.stack([audio_feats, video_feats], dim=1)

        if self.graph_enable_spatial:
            assert self.spatial_net
            # 从数据中提取空间关系特征
            spatial_grayscale_imgs = batch_create_spatial_grayscale(
                edge_attr_info[:, :2],
                self.spatial_grayscale_width,
                self.spatial_grayscale_height,
                3,
            )
            spatial_feats = []
            for i in range(0, spatial_grayscale_imgs.size(0), self.spatial_mini_batch):
                # 避免留下单个节点，导致无法 batch norm
                mini_spatial_grayscale_imgs = spatial_grayscale_imgs[
                    i : i + self.spatial_mini_batch
                ]
                if mini_spatial_grayscale_imgs.size(0) == 1:
                    mini_spatial_grayscale_imgs = torch.cat(
                        [mini_spatial_grayscale_imgs, mini_spatial_grayscale_imgs],
                        dim=0,
                    )
                    spatial_feats.append(
                        self.spatial_net(mini_spatial_grayscale_imgs)[0].unsqueeze(0)
                    )
                else:
                    spatial_feats.append(self.spatial_net(mini_spatial_grayscale_imgs))
            edge_attr = torch.cat(spatial_feats, dim=0)
            edge_attr = self.spatial_bn(edge_attr)
        else:
            # 权重全为 1，代表没有空间关系
            edge_attr = torch.ones(
                edge_attr_info.size(0),
                self.spatial_feature_dim,
                device=edge_attr_info.device,
            )
        data.edge_attr = edge_attr

        graph_out = self.graph_net(data)

        return graph_out, audio_out, video_out, vf_a_emb, vf_v_emb


############### 以下是模型的加载权重 ###############

def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)


############### 以下是模型的工厂函数 ###############


def get_backbone(
    encoder_type: str,
    graph_type: str,
    encoder_enable_vf: bool,
    graph_enable_spatial: bool,
    encoder_enable_grad: bool = False,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    spatial_pretrained_weights=None,
    encoder_train_weights=None,
    train_weights=None,
):
    vf_emb_dim = 128
    encoder, a_feature_dim, v_feature_dim = get_encoder(
        encoder_type,
        encoder_enable_vf,
        encoder_enable_grad,
        video_pretrained_weigths,
        audio_pretrained_weights,
        encoder_train_weights,
    )
    spatial_feature_dim = 64
    spatial_net = (
        # get_spatial_mobilenet_net(spatial_feature_dim, spatial_pretrained_weights)
        get_spatial_net(spatial_feature_dim)
        if graph_enable_spatial
        else None
    )
    graph_net = get_graph(
        graph_type,
        a_feature_dim,
        v_feature_dim,
        vf_emb_dim,
        spatial_feature_dim,
    )
    model = MyModel(
        encoder,
        graph_net,
        encoder_enable_vf,
        graph_enable_spatial,
        spatial_net,
        spatial_feature_dim=spatial_feature_dim,
    )

    # 加载模型权重
    if train_weights is not None:
        _load_weights_into_model(model, train_weights)
        print("loaded model weights")
        model.eval()
        # 先加载整个的，如果有单独的 encoder 的权重，再覆盖一次 encoder 的权重
        if encoder_train_weights is not None:
            _load_weights_into_model(encoder, encoder_train_weights)
            print("loaded encoder weights")

    return model, encoder


def get_encoder(
    encoder_type: str,
    encoder_enable_vf: bool,
    encoder_enable_grad: bool,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    encoder_train_weights=None,
):
    if encoder_type == "R3D18":
        encoder, a_feature_dim, v_feature_dim = get_resnet_encoder(
            "R3D18",
            encoder_enable_vf,
            encoder_enable_grad,
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "R3D50":
        encoder, a_feature_dim, v_feature_dim = get_resnet_encoder(
            "R3D50",
            encoder_enable_vf,
            encoder_enable_grad,
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "LIGHT":
        encoder, a_feature_dim, v_feature_dim = get_light_encoder(
            encoder_train_weights, encoder_enable_vf
        )
    elif encoder_type == "RES18_TSM":
        encoder, a_feature_dim, v_feature_dim = get_resnet_tsm_encoder(
            "resnet18",
            encoder_enable_vf,
            encoder_enable_grad,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RES50_TSM":
        encoder, a_feature_dim, v_feature_dim = get_resnet_tsm_encoder(
            "resnet50",
            encoder_enable_vf,
            encoder_enable_grad,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RX3D50":
        encoder, a_feature_dim, v_feature_dim = get_resnext_encoder(
            "RESNEXT50",
            encoder_enable_vf,
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RX3D101":
        encoder, a_feature_dim, v_feature_dim = get_resnext_encoder(
            "RESNEXT101",
            encoder_enable_vf,
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RX3D152":
        encoder, a_feature_dim, v_feature_dim = get_resnext_encoder(
            "RESNEXT152",
            encoder_enable_vf,
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "MB3DV1":
        encoder, a_feature_dim, v_feature_dim = get_mobilev1_encoder(
            encoder_enable_vf,
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "MB3DV2":
        encoder, a_feature_dim, v_feature_dim = get_mobilev2_encoder(
            encoder_enable_vf,
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder, a_feature_dim, v_feature_dim


def get_graph(
    graph_type: str,
    a_feature_dim: int,
    v_feature_dim: int,
    vf_emb_dim: int,
    edge_attr_dim: int,
):
    if graph_type == "GraphAllEdgeWeightNet":
        graph_net = GraphAllEdgeWeightNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphAllEdgeNet":
        graph_net = GraphAllEdgeNet(a_feature_dim, v_feature_dim, vf_emb_dim, 128)
    elif graph_type == "GraphGatEdgeNet":
        graph_net = GraphGatEdgeNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim, is_gatv2=True
        )
    elif graph_type == "GraphGatedEdgeNet":
        graph_net = GraphGatedEdgeNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphGinEdgeNet":
        graph_net = GraphGinEdgeNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphTransformerEdgeNet":
        graph_net = GraphTransformerEdgeNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphRnnGruNet":
        graph_net = GraphRnnNet(
            "GRU", a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphRnnLstmNet":
        graph_net = GraphRnnNet(
            "LSTM", a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return graph_net
