#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.parameter
from torch.nn import functional as F

from active_speaker_detection.models.vf_extract.vfal_sl_encoder import VfalSlEncoder
from active_speaker_detection.utils.vf_util import cosine_similarity

from .resnet.shared_2d import BasicBlock2D, conv1x1
from .resnet.shared_3d import BasicBlock3D, Bottleneck3D, conv1x1x1, get_inplanes


class ResnetTwoStreamNet(nn.Module):
    """
    两个流的网络，包含音频编码器和视频编码器
    """

    def __init__(
        self,
        # 音频编码器参数
        args_2d: Tuple[nn.Module, List[int], bool, int, int, List[bool], nn.Module],
        # 视频编码器参数
        args_3d: Tuple[
            nn.Module, List[int], List[int], int, int, int, bool, str, float
        ],
        encoder_enable_vf: bool = True,
        encoder_enable_grad: bool = False,
    ):
        super().__init__()

        (
            block_2d,  # 2D 音频基础块
            layers_2d,  # 2D 音频层的每层块数，[2, 2, 2, 2]
            zero_init_residual,
            groups_2d,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer_2d,  # 归一化层
        ) = args_2d
        (
            block_3d,  # 3D 视频基础块
            layers_3d,  # 3D 视频层的每层块数，[2, 2, 2, 2]
            block_inplanes_3d,  # 3D 视频基础块的输入通道数列表
            n_input_channels,  # 视频输入通道数
            conv1_t_size,  # 视频输入卷积层的时间维度卷积核大小
            conv1_t_stride,  # 视频输入卷积层的时间维度卷积步长
            no_max_pool,  # 是否不使用最大池化
            shortcut_type,  # 残差连接类型, A 表示使用 1x1 卷积层，B 表示使用 3x3 卷积层
            widen_factor,  # 宽度因子，对 block_inplanes_3d 的每个元素乘以宽度因子
        ) = args_3d

        if norm_layer_2d is None:
            # 默认归一化层
            norm_layer_2d = nn.BatchNorm2d
        self._norm_layer_2d = norm_layer_2d

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        # 音频流
        self.inplanes_2d = 64
        self.dilation_2d = 1
        self.groups_2d = groups_2d
        self.base_width = width_per_group
        self.audio_conv1 = nn.Conv2d(
            1, self.inplanes_2d, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.a_bn1 = norm_layer_2d(self.inplanes_2d)
        self.a_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a_layer1 = self._make_layer_2D(block_2d, 64, layers_2d[0])
        self.a_layer2 = self._make_layer_2D(
            block_2d,
            128,
            layers_2d[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.a_layer3 = self._make_layer_2D(
            block_2d,
            256,
            layers_2d[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.a_layer4 = self._make_layer_2D(
            block_2d,
            512,
            layers_2d[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.a_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 视频流
        block_inplanes = [int(x * widen_factor) for x in block_inplanes_3d]
        self.in_planes_3d = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.v_conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes_3d,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.v_bn1 = nn.BatchNorm3d(self.in_planes_3d)
        self.v_maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)
        )
        self.v_layer1 = self._make_layer_3D(
            block_3d, block_inplanes[0], layers_3d[0], shortcut_type
        )
        self.v_layer2 = self._make_layer_3D(
            block_3d, block_inplanes[1], layers_3d[1], shortcut_type, stride=2
        )
        self.v_layer3 = self._make_layer_3D(
            block_3d, block_inplanes[2], layers_3d[2], shortcut_type, stride=2
        )
        self.v_layer4 = self._make_layer_3D(
            block_3d, block_inplanes[3], layers_3d[3], shortcut_type, stride=2
        )
        self.v_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if not encoder_enable_grad:
            # 冻结上面的层
            for param in self.parameters():
                param.requires_grad = False

        # 音脸分支
        self.encoder_enable_vf = encoder_enable_vf
        self.vf_layer = VfalSlEncoder(
            voice_size=512, face_size=512, embedding_size=128, shared=False
        )

        # 共享
        self.relu = nn.ReLU(inplace=True)

        # 降维，降到节点特征的维度 128
        self.reduction_a = nn.Linear(512 * block_2d.expansion, 128)  # type: ignore
        self.reduction_v = nn.Linear(512 * block_3d.expansion, 128)  # type: ignore

        # 分类器
        self.fc_a = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 2)
        self.fc_av = nn.Linear(128 * 2, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 使用Kaiming初始化（nn.init.kaiming_normal_）来初始化权重
                # Kaiming初始化是一种特殊的方法，它可以在ReLU激活函数中保持方差不变。
                # 这里的mode="fan_out"表示权重的标准差是根据每个过滤器的输出连接数（而不是输入连接数）来计算的。
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                # 使用常数初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化（nn.init.kaiming_normal_）来初始化权重
                # Kaiming初始化是一种特殊的方法，它可以在ReLU激活函数中保持方差不变。
                # 这里的mode="fan_out"表示权重的标准差是根据每个过滤器的输出连接数（而不是输入连接数）来计算的。
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 使用常数初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_2D(
        self, block: nn.Module, planes: int, blocks: int, stride=1, dilate=False
    ):
        """构建 2D 网络的层，编码器层
        :param block: 基础块
        :param planes: 输出通道数
        :param blocks: 块数
        :param stride: 卷积步长
        """
        norm_layer = self._norm_layer_2d
        # 下采样块
        downsample: Optional[nn.Module] = None
        previous_dilation = self.dilation_2d
        if dilate:
            self.dilation_2d *= stride
            stride = 1
        if stride != 1 or self.inplanes_2d != planes * block.expansion:  # type: ignore
            downsample = nn.Sequential(
                conv1x1(self.inplanes_2d, planes * block.expansion, stride),  # type: ignore
                norm_layer(planes * block.expansion),  # type: ignore
            )

        # 构建层
        layers = []
        layers.append(
            block(
                self.inplanes_2d,
                planes,
                stride,
                downsample,
                self.groups_2d,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes_2d = planes * block.expansion  # type: ignore
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes_2d,
                    planes,
                    groups=self.groups_2d,
                    base_width=self.base_width,
                    dilation=self.dilation_2d,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):  # type: ignore
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer_3D(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes_3d != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes_3d, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes_3d,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes_3d = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes_3d, planes))

        return nn.Sequential(*layers)

    def forward_audio(self, a):
        a = self.audio_conv1(a)
        a = self.a_bn1(a)
        a = self.relu(a)
        a = self.a_maxpool(a)

        a = self.a_layer1(a)
        a = self.a_layer2(a)
        a = self.a_layer3(a)
        a = self.a_layer4(a)

        a = self.a_avgpool(a)
        a = a.reshape(a.size(0), -1)
        return a

    def forward_video(self, v):
        v = self.v_conv1(v)
        v = self.v_bn1(v)
        v = self.relu(v)
        if not self.no_max_pool:
            v = self.v_maxpool(v)

        v = self.v_layer1(v)
        v = self.v_layer2(v)
        v = self.v_layer3(v)
        v = self.v_layer4(v)

        v = self.v_avgpool(v)
        v = v.reshape(v.size(0), -1)
        return v

    def forward(self, a, v):
        """
        :param a: (B, C, 13, T)
        :param v: (B, C, T, H, W)
        """
        # 音频和视频的特征提取
        audio_feats = self.forward_audio(a)
        video_feats = self.forward_video(v)

        # 降维，降到节点特征的维度 128
        audio_emb = self.relu(self.reduction_a(audio_feats))
        video_emb = self.relu(self.reduction_v(video_feats))

        if self.encoder_enable_vf:
            # 音脸分支
            vf_a_emb, vf_v_emb = self.vf_layer(audio_feats, video_feats)

            # sim 的维度是 (B, )
            sim = cosine_similarity(vf_a_emb, vf_v_emb)
            audio_emb = audio_emb * sim.unsqueeze(1)

            audio_out, video_out, av_out = (
                self.fc_a(audio_emb),
                self.fc_v(video_emb),
                self.fc_av(torch.cat([audio_emb, video_emb], dim=1)),
            )

            return (
                audio_feats,
                video_feats,
                audio_out,
                video_out,
                av_out,
                vf_a_emb,
                vf_v_emb,
            )
        else:
            audio_out, video_out, av_out = (
                self.fc_a(audio_emb),
                self.fc_v(video_emb),
                self.fc_av(torch.cat([audio_emb, video_emb], dim=1)),
            )

            return (
                audio_feats,
                video_feats,
                audio_out,
                video_out,
                av_out,
                None,
                None,
            )


############### 以下是模型的加载权重 ###############


def _load_video_weights_into_model(model: nn.Module, ws_file):
    """加载视频预训练权重"""
    resnet_state_dict = torch.load(ws_file)["state_dict"]

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():
        if "v_" + name in own_state:
            own_state["v_" + name].copy_(param)
        else:
            print("No video assignation for ", name)

    print("loaded video from resnet")
    return


def _load_audio_weights_into_model(model: nn.Module, audio_pretrained_weights):
    """加载音频预训练权重，这里使用了 resnet18 的预训练权重"""
    resnet_state_dict = torch.load(audio_pretrained_weights)

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():
        if "a_" + name in own_state:
            own_state["a_" + name].copy_(param)
        else:
            print("No audio assignation for ", name)

    # Audio initial Ws
    conv1_weights = resnet_state_dict["conv1.weight"]
    avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
    own_state["audio_conv1.weight"].copy_(avgWs)

    print("loaded audio from resnet")
    return


def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)
    print("loaded encoder weights")


############### 以下是模型的工厂函数 ###############


def get_resnet_encoder(
    type: str,
    encoder_enable_vf: bool,
    encoder_enable_grad: bool,
    video_pretrained_weigths=None,
    audio_pretrained_weights=None,
    encoder_train_weights=None,
):
    if type == "R3D18":
        args_2d = BasicBlock2D, [2, 2, 2, 2], False, 1, 64, None, None
        args_3d = BasicBlock3D, [2, 2, 2, 2], get_inplanes(), 3, 7, 1, False, "B", 1.0
        encoder = ResnetTwoStreamNet(args_2d, args_3d, encoder_enable_vf, encoder_enable_grad)  # type: ignore
        if video_pretrained_weigths is not None:
            _load_video_weights_into_model(encoder, video_pretrained_weigths)
        if audio_pretrained_weights is not None:
            _load_audio_weights_into_model(encoder, audio_pretrained_weights)
        if encoder_train_weights is not None:
            _load_weights_into_model(encoder, encoder_train_weights)
            encoder.eval()
        return encoder, 512 * BasicBlock2D.expansion, 512 * BasicBlock3D.expansion
    elif type == "R3D50":
        args_2d = BasicBlock2D, [2, 2, 2, 2], False, 1, 64, None, None
        args_3d = Bottleneck3D, [3, 4, 6, 3], get_inplanes(), 3, 7, 1, False, "B", 1.0
        encoder = ResnetTwoStreamNet(args_2d, args_3d, encoder_enable_vf, encoder_enable_grad)  # type: ignore
        if video_pretrained_weigths is not None:
            _load_video_weights_into_model(encoder, video_pretrained_weigths)
        if audio_pretrained_weights is not None:
            _load_audio_weights_into_model(encoder, audio_pretrained_weights)
        if encoder_train_weights is not None:
            _load_weights_into_model(encoder, encoder_train_weights)
            encoder.eval()
        return encoder, 512 * BasicBlock2D.expansion, 512 * Bottleneck3D.expansion
    else:
        raise ValueError("Unknown type")
