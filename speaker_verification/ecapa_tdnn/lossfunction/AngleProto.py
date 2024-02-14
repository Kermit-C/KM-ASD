#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-14 09:20:42
"""

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.ecapa_tdnn.utils.acc import accuracy


class AngleProto(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(AngleProto, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]

        stepsize = out_anchor.size()[0]
        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda()
        nloss = self.criterion(cos_sim_matrix, label)
        # print("lossC:", self.criterion(cos_sim_matrix, label))
        prec1 = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1
