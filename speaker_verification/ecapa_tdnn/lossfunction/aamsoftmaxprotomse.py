import torch
import torch.nn as nn

from . import AngleProtoMSE as AngleProtoMSE
from . import aamsoftmax as aamsoftmax


class AamSoftmaxProtoMSE(nn.Module):

    def __init__(self, nOut, nClasses, margin, scale, w = 10):
        super(AamSoftmaxProtoMSE, self).__init__()

        self.test_normalize = True
        self.w = w
        self.aamsoftmax = aamsoftmax.AamSoftmax(nOut, nClasses, margin, scale)
        self.angleprotomse = AngleProtoMSE()

        print('Initialised AamSoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.aamsoftmax(x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))

        nlossP, _ = self.angleprotomse(x, None)
        # print("lossP:", nlossP, "nlossS:", nlossS)
        # lossP:0.6678 nlossS:13.6913

        return nlossS + self.w * nlossP, prec1
