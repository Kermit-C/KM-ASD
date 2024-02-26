#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import torch
import torch.nn as nn

from active_speaker_detection.models.vfal.vfal_sl_encoder import VfalSlEncoder


class AudioBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AudioBlock, self).__init__()

        self.relu = nn.ReLU()

        self.m_3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False
        )
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False
        )
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.m_5 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False
        )
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False
        )
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.last = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=False
        )
        self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

        return x


class VisualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=False):
        super(VisualBlock, self).__init__()

        self.relu = nn.ReLU()

        if is_down:
            # 空间 H * W 上的卷积，实质上是 2D 卷积
            self.s_3 = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            )
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            # 时间 T 上的卷积，实质上是 1D 卷积
            self.t_3 = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            )
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

            self.s_5 = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
                bias=False,
            )
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                bias=False,
            )
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        else:
            self.s_3 = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            )
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            )
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

            self.s_5 = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 5, 5),
                padding=(0, 2, 2),
                bias=False,
            )
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(5, 1, 1),
                padding=(2, 0, 0),
                bias=False,
            )
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.last = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.bn_last = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):

        x_3 = self.relu(self.bn_s_3(self.s_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_s_5(self.s_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5

        x = self.relu(self.bn_last(self.last(x)))

        return x


class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()

        self.block1 = VisualBlock(3, 32, is_down=True)
        self.pool1 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.block2 = VisualBlock(32, 64)
        self.pool2 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.block3 = VisualBlock(64, 128)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()

    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = x.transpose(1, 2)
        B, T, C, W, H = x.shape
        x = x.reshape(B * T, C, W, H)

        x = self.maxpool(x)

        x = x.view(B, T, C)

        return x

    def __init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.block1 = AudioBlock(1, 32)
        self.pool1 = nn.MaxPool3d(
            kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1)
        )

        self.block2 = AudioBlock(32, 64)
        self.pool2 = nn.MaxPool3d(
            kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1)
        )

        self.block3 = AudioBlock(64, 128)

        self.__init_weight()

    def forward(self, x):
        # x: (B, C, 13, T)
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim=2, keepdim=True)
        x = x.squeeze(2).transpose(1, 2)
        # x: (B, T, C)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LightTwoStreamNet(nn.Module):

    def __init__(self, encoder_enable_vf: bool):
        super(LightTwoStreamNet, self).__init__()

        self.audio_encoder = AudioEncoder()
        self.visual_encoder = VisualEncoder()

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # 音脸分支
        self.encoder_enable_vf = encoder_enable_vf
        self.vf_layer = VfalSlEncoder(
            voice_size=128, face_size=128, embedding_size=128, shared=True
        )

        # 分类器
        self.fc_a = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 2)
        self.fc_av = nn.Linear(128 * 2, 2)

        self.__init_weight()

    def forward_visual(self, x):
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visual_encoder(x)
        return x

    def forward_audio(self, x):
        x = self.audio_encoder(x)
        return x

    def forward(self, audio_data, video_data):
        """
        :param a: (B, C, 13, T)
        :param v: (B, C, T, H, W)
        """
        # audio_embed: (B, T, C)
        audio_embed = self.forward_audio(audio_data)
        # audio_embed: (B, T, C)
        visual_embed = self.forward_visual(video_data)

        audio_embed = self.max_pool(audio_embed.transpose(1, 2)).squeeze(2)
        visual_embed = self.max_pool(visual_embed.transpose(1, 2)).squeeze(2)

        if self.encoder_enable_vf:
            # 音脸分支
            vf_a_emb, vf_v_emb = self.vf_layer(audio_embed, visual_embed)

            audio_embed += vf_a_emb
            visual_embed += vf_v_emb

            audio_out, video_out, av_out = (
                self.fc_a(audio_embed),
                self.fc_v(visual_embed),
                self.fc_av(torch.cat([audio_embed, visual_embed], dim=1)),
            )

            # audio_embed: (B, C), visual_embed: (B, C)
            return (
                audio_embed,
                visual_embed,
                audio_out,
                video_out,
                av_out,
                vf_a_emb,
                vf_v_emb,
            )
        else:
            audio_out, video_out, av_out = (
                self.fc_a(audio_embed),
                self.fc_v(visual_embed),
                self.fc_av(torch.cat([audio_embed, visual_embed], dim=1)),
            )

            # audio_embed: (B, C), visual_embed: (B, C)
            return audio_embed, visual_embed, audio_out, video_out, av_out, None, None

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()


############### 以下是模型的加载权重 ###############


def _load_weights_into_model(model: nn.Module, ws_file):
    """加载训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)
    return


############### 以下是模型的工厂函数 ###############


def get_light_encoder(
    encoder_train_weights=None,
    encoder_enable_vf=True,
):
    model = LightTwoStreamNet(encoder_enable_vf)
    if encoder_train_weights:
        _load_weights_into_model(model, encoder_train_weights)
        model.eval()
    return model
