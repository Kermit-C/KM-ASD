#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-24 12:38:39
"""


import torch
import torch.nn as nn
import torch.nn.parameter
from torch.nn import functional as F

from active_speaker_detection.models.tsm.temporal_shift_layout import TemporalShift
from active_speaker_detection.models.vf_extract.vfal_sl_encoder import VfalSlEncoder
from active_speaker_detection.utils.vf_util import cosine_similarity

from .resnet.shared_2d import BasicBlock2D, Bottleneck2D, conv1x1


class TwoStreamResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        encoder_enable_vf=True,
        encoder_enable_grad=False,
    ):
        super(TwoStreamResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # audio stream
        self.inplanes = 64
        self.audio_conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.a_bn1 = norm_layer(self.inplanes)
        self.a_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a_layer1 = self._make_layer(block, 64, layers[0])
        self.a_layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.a_layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.a_layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        # visual stream
        self.inplanes = 64
        self.video_conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.v_bn1 = norm_layer(self.inplanes)
        self.v_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.v_layer1 = self._make_layer(block, 64, layers[0])
        self.v_layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.v_layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.v_layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        if not encoder_enable_grad:
            # 冻结上面参数
            for param in self.parameters():
                param.requires_grad = False

        # 音脸分支
        self.encoder_enable_vf = encoder_enable_vf
        self.vf_layer = VfalSlEncoder(
            voice_size=512 * block.expansion,
            face_size=512 * block.expansion,
            embedding_size=128,
            shared=False,
        )

        self.tsm = TemporalShift(n_div=8)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_128_a = nn.Linear(512 * block.expansion, 128)
        self.fc_128_v = nn.Linear(512 * block.expansion, 128)

        # 分类器
        self.fc_a = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 2)
        self.fc_av = nn.Linear(128 * 2, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # this improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2D):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr]
                elif isinstance(m, BasicBlock2D):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[union-attr]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, a, v):
        """
        :param a: (B, C, 13, T)
        :param v: (B, C, T, H, W)
        """
        a = self.audio_conv1(a)
        a = self.a_bn1(a)
        a = self.relu(a)
        a = self.maxpool(a)

        a = self.a_layer1(a)
        a = self.a_layer2(a)
        a = self.a_layer3(a)
        a = self.a_layer4(a)
        a = self.avgpool(a)
        a = a.squeeze(-1).squeeze(-1)

        v = v.transpose(1, 2).contiguous()

        b, t, c, h, w = v.size()
        v = v.view(b * t, c, h, w)
        v = self.video_conv1(v)
        v = self.v_bn1(v)
        v = self.relu(v)
        v = self.maxpool(v)

        v = self.tsm(v, t)
        v = self.v_layer1(v)
        v = self.tsm(v, t)
        v = self.v_layer2(v)
        v = self.tsm(v, t)
        v = self.v_layer3(v)
        v = self.tsm(v, t)
        v = self.v_layer4(v)
        v = self.avgpool(v)
        v = v.view(b, t, v.size(1), v.size(2), v.size(3))

        v = v.transpose(1, 2).contiguous()
        v = v.squeeze(-1).squeeze(-1)
        v = v.mean(2)

        # 降维到 128
        a_emb = self.fc_128_a(a)
        a_emb = self.relu(a_emb)
        v_emb = self.fc_128_v(v)
        v_emb = self.relu(v_emb)

        if self.encoder_enable_vf:
            # 音脸分支
            vf_a_emb, vf_v_emb = self.vf_layer(a, v)

            # sim 的维度是 (B, )
            sim = cosine_similarity(vf_a_emb, vf_v_emb)

            audio_out, video_out, av_out = (
                self.fc_a(a_emb),
                self.fc_v(v_emb),
                self.fc_av(torch.cat([a_emb * sim.unsqueeze(1), v_emb], dim=1)),
            )

            return a, v, audio_out, video_out, av_out, vf_a_emb, vf_v_emb
        else:
            audio_out, video_out, av_out = (
                self.fc_a(a_emb),
                self.fc_v(v_emb),
                self.fc_av(torch.cat([a_emb, v_emb], dim=1)),
            )

            return a, v, audio_out, video_out, av_out, None, None


############### 以下是模型的加载权重 ###############


def _load_weights_into_two_stream_resnet(model, pretrained_weights):
    resnet_state_dict = torch.load(pretrained_weights)

    own_state = model.state_dict()
    for name, param in resnet_state_dict.items():
        if "v_" + name in own_state:
            own_state["v_" + name].copy_(param)
        if "a_" + name in own_state:
            own_state["a_" + name].copy_(param)
        if "v_" + name not in own_state and "a_" + name not in own_state:
            print("No assignation for ", name)

    conv1_weights = resnet_state_dict["conv1.weight"]
    own_state["video_conv1.weight"].copy_(conv1_weights)

    avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
    own_state["audio_conv1.weight"].copy_(avgWs)

    print("loaded ws from resnet")
    return model


def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)
    print("loaded encoder weights")


############### 以下是模型的工厂函数 ###############


def get_resnet_tsm_encoder(
    type: str,
    encoder_enable_vf: bool,
    encoder_enable_grad: bool = False,
    pretrained_weigths=None,
    encoder_train_weights=None,
):
    if type == "resnet18":
        block, layers = BasicBlock2D, [2, 2, 2, 2]
        model = TwoStreamResNet(
            block,
            layers,
            encoder_enable_vf=encoder_enable_vf,
            encoder_enable_grad=encoder_enable_grad,
        )
        if pretrained_weigths is not None:
            model = _load_weights_into_two_stream_resnet(model, pretrained_weigths)
        if encoder_train_weights is not None:
            _load_weights_into_model(model, encoder_train_weights)
        return model, 512 * block.expansion, 512 * block.expansion
    elif type == "resnet50":
        block, layers = Bottleneck2D, [3, 4, 6, 3]
        model = TwoStreamResNet(
            block,
            layers,
            encoder_enable_vf=encoder_enable_vf,
            encoder_enable_grad=encoder_enable_grad,
        )
        if pretrained_weigths is not None:
            model = _load_weights_into_two_stream_resnet(model, pretrained_weigths)
        if encoder_train_weights is not None:
            _load_weights_into_model(model, encoder_train_weights)
        return model, 512 * block.expansion, 512 * block.expansion
    else:
        raise ValueError("Unknown resnet type: {}".format(type))
