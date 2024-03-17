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

from ..utils.spatial_grayscale_util import batch_create_spatial_grayscale
from ..utils.vf_util import cosine_similarity
from .graph_all_edge_net import GraphAllEdgeNet
from .graph_easee_net import GraphEaseeNet
from .graph_fc_net import GraphFcNet
from .graph_gat_2s_net import GraphGatTwoStreamNet
from .graph_gat_edge_net import GraphGatEdgeNet
from .graph_gated_edge_net import GraphGatedEdgeNet
from .graph_gin_edge_net import GraphGinEdgeNet
from .graph_km_2s_edge_net import GraphKmTwoStreamEdgeNet
from .graph_km_2s_net import GraphKmTwoStreamNet
from .graph_km_edge_net import GraphKmEdgeNet
from .graph_rnn_net import GraphRnnNet
from .graph_transformer_edge_net import GraphTransformerEdgeNet
from .spatial_mobilenet_net import get_spatial_mobilenet_net
from .spatial_net import get_spatial_net
from .two_stream_light_net import get_light_encoder
from .two_stream_mobilev1_net import get_mobilev1_encoder
from .two_stream_mobilev2_net import get_mobilev2_encoder
from .two_stream_resnet_net import get_resnet_encoder
from .two_stream_resnet_tsm_net import get_resnet_tsm_encoder
from .two_stream_resnext_net import get_resnext_encoder
from .vf_extract.vfal_sl_encoder import VfalSlEncoder


class MyModel(nn.Module):

    def __init__(
        self,
        encoder,
        graph_net,
        encoder_enable_vf: bool,
        graph_enable_spatial: bool,
        encoder_vf: Optional[VfalSlEncoder] = None,  # 为空则不使用音视频融合网络
        spatial_net: Optional[nn.Module] = None,  # 为空则不使用空间关系网络
        encoder_vf_emb_dim: int = 128,
        spatial_feature_dim: int = 64,
        spatial_grayscale_width: int = 32,
        spatial_grayscale_height: int = 32,
    ):
        super().__init__()

        self.encoder = encoder
        self.graph_net = graph_net

        self.encoder_enable_vf = encoder_enable_vf
        self.graph_enable_spatial = graph_enable_spatial

        self.encoder_vf = encoder_vf
        self.encoder_vf_emb_dim = encoder_vf_emb_dim

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
        x, edge_index, edge_attr_info = data.x, data.edge_index, data.edge_attr
        audio_node_mask = []
        audio_feature_idx_list = []
        if type(data.audio_node_mask[0]) == bool:
            audio_node_mask = data.audio_node_mask
            audio_feature_idx_list = data.audio_feature_idx_list
        else:
            # 批量数据平铺
            for mask, idx in zip(data.audio_node_mask, data.audio_feature_idx_list):
                audio_node_mask += mask
                audio_feature_idx_list += [i + len(audio_feature_idx_list) for i in idx]
        video_node_mask = [not mask for mask in audio_node_mask]

        if len(x.shape) > 3:
            # 从数据中提取音频和视频特征

            # encoder
            assert audio_size is not None
            encoder_audio_data = x[audio_node_mask][
                :, 0, 0, 0, : audio_size[1], : audio_size[2]
            ].unsqueeze(1)
            encoder_video_data = x[video_node_mask][:, 1, :, :, :, :]
            encoder_audio_feats, encoder_video_feats, *_ = self.encoder(
                encoder_audio_data, encoder_video_data
            )

            # 从 encoder 输出构造完整数据
            audio_feats = torch.zeros(
                x.size(0),
                encoder_audio_feats.size(1),
                dtype=encoder_audio_feats.dtype,
                device=encoder_audio_feats.device,
            )
            video_feats = torch.zeros(
                x.size(0),
                encoder_video_feats.size(1),
                dtype=encoder_video_feats.dtype,
                device=encoder_video_feats.device,
            )
            audio_feats[audio_node_mask] = encoder_audio_feats
            audio_feats = audio_feats[audio_feature_idx_list]  # 填充非纯音频节点的部分
            video_feats[video_node_mask] = encoder_video_feats

            audio_feat_dim = audio_feats.size(1)
            video_feat_dim = video_feats.size(1)
            if self.encoder_enable_vf:
                vf_a_emb, vf_v_emb = self.encoder_vf(audio_feats, video_feats)  # type: ignore
                vf_a_emb_dim = vf_a_emb.size(1)
                vf_v_emb_dim = vf_v_emb.size(1)

                max_dim = max(
                    audio_feat_dim, video_feat_dim, vf_a_emb_dim, vf_v_emb_dim
                )
                # 维度不够的在后面填充 0
                audio_feats = F.pad(audio_feats, (0, max_dim - audio_feat_dim))
                video_feats = F.pad(video_feats, (0, max_dim - video_feat_dim))
                pad_vf_a_emb = F.pad(vf_a_emb, (0, max_dim - vf_a_emb_dim))
                pad_vf_v_emb = F.pad(vf_v_emb, (0, max_dim - vf_v_emb_dim))
                data.x = torch.stack(
                    [audio_feats, video_feats, pad_vf_a_emb, pad_vf_v_emb], dim=1
                )
            else:
                vf_a_emb, vf_v_emb = None, None
                max_dim = max(audio_feat_dim, video_feat_dim)
                audio_feats = F.pad(audio_feats, (0, max_dim - audio_feat_dim))
                video_feats = F.pad(video_feats, (0, max_dim - video_feat_dim))
                data.x = torch.stack([audio_feats, video_feats], dim=1)
        else:
            # 输入的就是 encoder 出来的特征
            if self.encoder_enable_vf:
                vf_a_emb = x[:, 2, : self.encoder_vf_emb_dim]
                vf_v_emb = x[:, 3, : self.encoder_vf_emb_dim]
            else:
                vf_a_emb, vf_v_emb = None, None
                audio_feats = x[:, 0, :]
                video_feats = x[:, 1, :]
                data.x = torch.stack([audio_feats, video_feats], dim=1)

        if self.graph_enable_spatial:
            assert self.spatial_net

            # 从数据中提取空间关系特征
            time_delta_rate = edge_attr_info[:, 3, 0]
            # 时间差比例作为空间关系的颜色深度
            spatial_grayscale_values = torch.exp(-time_delta_rate * 10) * 255
            spatial_grayscale_imgs = batch_create_spatial_grayscale(
                edge_attr_info[:, :2],
                self.spatial_grayscale_width,
                self.spatial_grayscale_height,
                spatial_grayscale_values,
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

            edge_attr = torch.zeros(
                edge_attr_info.size(0),
                self.spatial_feature_dim,
                device=edge_attr_info.device,
                dtype=x.dtype,
            )
            edge_attr[:] = torch.cat(spatial_feats, dim=0)

            edge_vertices_audio = edge_attr_info[:, 2, :2]
            edge_vertices_no_audio_mask = edge_vertices_audio.mean(dim=1) == 0
            edge_vertices_half1_audio_mask = (edge_vertices_audio[:, 0] == 1) & (
                edge_vertices_audio[:, 1] == 0
            )  # 从音到脸
            edge_vertices_half2_audio_mask = (edge_vertices_audio[:, 0] == 0) & (
                edge_vertices_audio[:, 1] == 1
            )  # 从脸到音
            edge_vertices_all_audio_mask = edge_vertices_audio.mean(dim=1) == 1
            if self.encoder_enable_vf:
                # 融合音脸关系相似度
                assert vf_a_emb is not None
                assert vf_v_emb is not None
                edge_half1_audio_index = edge_index[:, edge_vertices_half1_audio_mask]
                edge_half2_audio_index = edge_index[:, edge_vertices_half2_audio_mask]
                vf_a_emb_half1_audio = vf_a_emb[edge_half1_audio_index[0]]
                vf_v_emb_half1_audio = vf_v_emb[edge_half1_audio_index[1]]
                vf_v_emb_half2_audio = vf_v_emb[edge_half2_audio_index[0]]
                vf_a_emb_half2_audio = vf_a_emb[edge_half2_audio_index[1]]
                sim_half1 = cosine_similarity(
                    vf_a_emb_half1_audio, vf_v_emb_half1_audio
                )
                sim_half2 = cosine_similarity(
                    vf_a_emb_half2_audio, vf_v_emb_half2_audio
                )
                edge_attr[edge_vertices_half1_audio_mask] = edge_attr[
                    edge_vertices_half1_audio_mask
                ] * sim_half1.unsqueeze(1)
                edge_attr[edge_vertices_half2_audio_mask] = edge_attr[
                    edge_vertices_half2_audio_mask
                ] * sim_half2.unsqueeze(1)

            edge_attr = self.spatial_bn(edge_attr)
        else:
            edge_vertices_audio = edge_attr_info[:, 2, :2]
            # 权重全为 1，代表没有空间关系
            edge_attr = torch.ones(
                edge_attr_info.size(0),
                self.spatial_feature_dim,
                device=edge_attr_info.device,
            )

        data.edge_attr = edge_attr
        data.edge_vertices_audio = edge_vertices_audio
        data.edge_delta = edge_attr_info[:, 4, 0]
        data.edge_self = edge_attr_info[:, 5, 0]

        graph_out, graph_audio_out, graph_video_out = self.graph_net(data)

        return graph_out, graph_audio_out, graph_video_out, vf_a_emb, vf_v_emb


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
    encoder, a_feature_dim, v_feature_dim = get_encoder(
        encoder_type,
        encoder_enable_grad,
        video_pretrained_weigths,
        audio_pretrained_weights,
        encoder_train_weights,
    )
    vf_emb_dim = 128
    encoder_vf = VfalSlEncoder(
        voice_size=a_feature_dim,
        face_size=v_feature_dim,
        embedding_size=vf_emb_dim,
        shared=False,
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
        encoder_vf,
        spatial_net,
        encoder_vf_emb_dim=vf_emb_dim,
        spatial_feature_dim=spatial_feature_dim,
    )

    # 加载模型权重
    if train_weights is not None:
        _load_weights_into_model(model, train_weights)
        print("loaded model weights")
        # 先加载整个的，如果有单独的 encoder 的权重，再覆盖一次 encoder 的权重
        if encoder_train_weights is not None:
            _load_weights_into_model(encoder, encoder_train_weights)
            print("loaded encoder weights")

    return model, encoder, encoder_vf if encoder_enable_vf else None


def get_encoder(
    encoder_type: str,
    encoder_enable_grad: bool,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    encoder_train_weights=None,
):
    if encoder_type == "R3D18":
        encoder, a_feature_dim, v_feature_dim = get_resnet_encoder(
            "R3D18",
            encoder_enable_grad,
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "R3D50":
        encoder, a_feature_dim, v_feature_dim = get_resnet_encoder(
            "R3D50",
            encoder_enable_grad,
            video_pretrained_weigths,
            audio_pretrained_weights,
            encoder_train_weights,
        )
    elif encoder_type == "LIGHT":
        encoder, a_feature_dim, v_feature_dim = get_light_encoder(encoder_train_weights)
    elif encoder_type == "RES18_TSM":
        encoder, a_feature_dim, v_feature_dim = get_resnet_tsm_encoder(
            "resnet18",
            encoder_enable_grad,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RES50_TSM":
        encoder, a_feature_dim, v_feature_dim = get_resnet_tsm_encoder(
            "resnet50",
            encoder_enable_grad,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RX3D50":
        encoder, a_feature_dim, v_feature_dim = get_resnext_encoder(
            "RESNEXT50",
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RX3D101":
        encoder, a_feature_dim, v_feature_dim = get_resnext_encoder(
            "RESNEXT101",
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "RX3D152":
        encoder, a_feature_dim, v_feature_dim = get_resnext_encoder(
            "RESNEXT152",
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "MB3DV1":
        encoder, a_feature_dim, v_feature_dim = get_mobilev1_encoder(
            encoder_enable_grad,
            audio_pretrained_weights,
            video_pretrained_weigths,
            encoder_train_weights,
        )
    elif encoder_type == "MB3DV2":
        encoder, a_feature_dim, v_feature_dim = get_mobilev2_encoder(
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
    if graph_type == "GraphKmEdgeNet":
        graph_net = GraphKmEdgeNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphKmTwoStreamEdgeNet":
        graph_net = GraphKmTwoStreamEdgeNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphKmTwoStreamNet":
        graph_net = GraphKmTwoStreamNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )

    elif graph_type == "GraphEaseeNet":
        graph_net = GraphEaseeNet(a_feature_dim, v_feature_dim, vf_emb_dim, 128)
    elif graph_type == "GraphFcNet":
        graph_net = GraphFcNet(a_feature_dim, v_feature_dim, vf_emb_dim, 128)
    elif graph_type == "GraphRnnGruNet":
        graph_net = GraphRnnNet(
            "GRU", a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphRnnLstmNet":
        graph_net = GraphRnnNet(
            "LSTM", a_feature_dim, v_feature_dim, vf_emb_dim, 128, edge_attr_dim
        )
    elif graph_type == "GraphGatTwoStreamNet":
        graph_net = GraphGatTwoStreamNet(
            a_feature_dim, v_feature_dim, vf_emb_dim, 128, is_gatv2=True
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

    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    return graph_net
