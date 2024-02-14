#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-14 09:20:42
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.ecapa_tdnn.utils.acc import accuracy


class AamSoftmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, **kwargs):
        super(AamSoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        # w是类的中心
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f' % (self.m, self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # print("cosine:", cosine.shape)
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        # phi = cos(ø+m)
        phi = cosine * self.cos_m - sine * self.sin_m
        # print(self.cos_m)
        # print("phi:", phi.shape)
        # 确保加上margin后是单调的，比较可比较
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # scale的作用扩大正样本，缩小负样本
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
        # return loss


if __name__ == "__main__":
    x = torch.randn(32, 512)
    y = torch.randint(1000, size=(32,))
    print(x.shape, y.shape)
    loss = AamSoftmax(512, 1000)
    nloss, prec1 = loss(x, y)
    print(nloss, prec1)
