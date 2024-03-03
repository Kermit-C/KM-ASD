#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-03-02 14:03:54
"""


from functools import partial

import torch
import torch.nn as nn

from active_speaker_detection.models.vf_extract.vfal_sl_encoder import VfalSlEncoder
from active_speaker_detection.utils.vf_util import cosine_similarity

from .resnet.shared_2d import BasicBlock2D, Bottleneck2D, conv1x1
from .resnext.shared_3d import ResNeXtBottleneck, downsample_basic_block


class AudioBackbone(nn.Module):

    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(AudioBackbone, self).__init__()
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

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

    def forward(self, a):
        """
        :param a: (B, C, 13, T)
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

        return a


class VideoBackbone(nn.Module):

    def __init__(
        self, block, layers, shortcut_type="B", cardinality=32, num_classes=400
    ):
        self.inplanes = 64
        super(VideoBackbone, self).__init__()
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.layer1 = self._make_layer(
            block, 128, layers[0], shortcut_type, cardinality
        )
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2
        )
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2
        )
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2
        )
        self.final_conv_bn = nn.Sequential(
            nn.Conv3d(2048, 512, 1, 1, 0, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        self.avgpool_3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_conv_bn(x)

        x = self.avgpool_3d(x)
        x = x.reshape(x.size(0), -1)

        return x


class ResnextTwoStreamNet(nn.Module):
    def __init__(
        self, video_backbone, encoder_enable_vf=True, encoder_enable_grad=False
    ):
        super(ResnextTwoStreamNet, self).__init__()
        self.audio_backbone = AudioBackbone(BasicBlock2D, [2, 2, 2, 2])
        self.video_backbone = video_backbone

        if not encoder_enable_grad:
            # 冻结上面参数
            for param in self.parameters():
                param.requires_grad = False

        # 音脸分支
        self.encoder_enable_vf = encoder_enable_vf
        self.vf_layer = VfalSlEncoder(
            voice_size=512 * BasicBlock2D.expansion,
            face_size=512,
            embedding_size=128,
            shared=False,
        )

        self.relu = nn.ReLU(inplace=True)
        self.fc_128_a = nn.Linear(512 * BasicBlock2D.expansion, 128)
        self.fc_128_v = nn.Linear(512, 128)

        # 分类器
        self.fc_a = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 2)
        self.fc_av = nn.Linear(128 * 2, 2)

    def forward(self, a, v):
        """
        :param a: (B, C, 13, T)
        :param v: (B, C, T, H, W)
        """
        a = self.audio_backbone(a)
        v = self.video_backbone(v)

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
            a_emb = a_emb * sim.unsqueeze(1)

            audio_out, video_out, av_out = (
                self.fc_a(a_emb),
                self.fc_v(v_emb),
                self.fc_av(torch.cat([a_emb, v_emb], dim=1)),
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


def _load_audio_pretrained_weights_into_model(model: nn.Module, ws_file):
    """加载预训练权重"""
    resnet_state_dict = torch.load(ws_file)

    own_state = model.audio_backbone.state_dict()  # type: ignore
    for name, param in resnet_state_dict.items():
        if "a_" + name in own_state:
            own_state["a_" + name].copy_(param)
        else:
            print("No assignation for ", name)

    conv1_weights = resnet_state_dict["conv1.weight"]
    avgWs = torch.mean(conv1_weights, dim=1, keepdim=True)
    own_state["audio_conv1.weight"].copy_(avgWs)

    print("loaded audio weights from resnet")


def _load_video_pretrained_weights_into_model(model: nn.Module, model_path):
    """加载预训练权重"""
    model = model.video_backbone  # type: ignore

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # print("loaded {}, epoch {}".format(model_path, checkpoint["epoch"]))
    state_dict_ = checkpoint["state_dict"]
    # state_dict_ = checkpoint
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            print("Drop parameter {}.".format(k))
    for k in model_state_dict:
        # num_batches_tracked need to be filtered out (a version problem
        # see https://stackoverflow.com/questions/53678133/load-pytorch-model-from-0-4-1-to-0-4-0)
        if not (k in state_dict) and "num_batches_tracked" not in k:
            print("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=False)  # type: ignore
    print("loaded video weights from 3d-resnext")


def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)  # type: ignore
    print("loaded encoder weights")


############### 以下是模型的工厂函数 ###############


def get_resnext_encoder(
    type: str,
    encoder_enable_vf: bool,
    encoder_enable_grad: bool = False,
    audio_pretrained_weigths=None,
    video_pretrained_weigths=None,
    encoder_train_weights=None,
):
    if type == "RESNEXT50":
        video_backbone = get_resnext50_video_backbone()
    elif type == "RESNEXT101":
        video_backbone = get_resnext101_video_backbone()
    elif type == "RESNEXT152":
        video_backbone = get_resnext152_video_backbone()
    else:
        raise ValueError(
            "type must be one of ['RESNEXT50', 'RESNEXT101', 'RESNEXT152']"
        )
    model = ResnextTwoStreamNet(video_backbone, encoder_enable_vf, encoder_enable_grad)

    if audio_pretrained_weigths is not None:
        _load_audio_pretrained_weights_into_model(model, audio_pretrained_weigths)
    if video_pretrained_weigths is not None:
        _load_video_pretrained_weights_into_model(model, video_pretrained_weigths)
    if encoder_train_weights is not None:
        _load_weights_into_model(model, encoder_train_weights)

    return model, 512 * BasicBlock2D.expansion, 512


def get_resnext50_video_backbone(**kwargs):
    """Constructs a ResNet-50 model."""
    model = VideoBackbone(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def get_resnext101_video_backbone(**kwargs):
    """Constructs a ResNet-101 model."""
    model = VideoBackbone(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def get_resnext152_video_backbone(**kwargs):
    """Constructs a ResNet-101 model."""
    model = VideoBackbone(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
