#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-28 15:47:32
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..mobilenet.shared import ConvNormActivation, InvertedResidual, _make_divisible


class SpatialMobileNet(nn.Module):

    def __init__(
        self,
        feature_dim: int = 128,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[list[list[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(SpatialMobileNet, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        features: list[nn.Module] = [
            ConvNormActivation(
                3,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        features.append(
            ConvNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )
        self.features = nn.Sequential(*features)

        # 将 mobilenet classifier 替换成降维层
        self.reduction = nn.Linear(self.last_channel, feature_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.reduction(x)
        return x


############### 以下是模型的加载权重 ###############


def _load_pretrained_weights_into_model(model: nn.Module, ws_file):
    """加载预训练权重"""
    model.load_state_dict(torch.load(ws_file), strict=False)  # type: ignore
    print("loaded spatial pretrained weights from mobilenet")


############### 以下是模型的工厂函数 ###############


def get_spatial_mobilenet_net(
    feature_dim: int,  # 输出特征维度
    spatial_pretrained_weights=None,
):
    model = SpatialMobileNet(feature_dim)
    if spatial_pretrained_weights is not None:
        _load_pretrained_weights_into_model(model, spatial_pretrained_weights)
    return model
