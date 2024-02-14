import torch
import torch.nn as nn

from . import AngleProto2 as AngleProto2
from . import aamsoftmax as aamsoftmax


class AamSoftmaxProto2(nn.Module):

    def __init__(self, nOut, nClasses, margin, scale, w=1):
        super(AamSoftmaxProto2, self).__init__()

        self.test_normalize = True
        self.w = w
        self.aamsoftmax = aamsoftmax.AamSoftmax(nOut, nClasses, margin, scale)
        self.angleproto = AngleProto2()

        print('Initialised AamSoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.aamsoftmax(x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))

        nlossP, _ = self.angleproto(x, label)
        # print("lossP:", nlossP, "nlossS:", nlossS)
        # print(nlossS + self.w * nlossP, self.w)
        # lossP:0.6678 nlossS:13.6913

        return nlossS + self.w * nlossP, prec1


if __name__ == "__main__":
    x = torch.randn(32, 2, 512)
    y = torch.randint(1000, size=(32,))
    print(x.shape, y.shape)
    loss = AamSoftmaxProto2(512, 1000, 0.2, 30)
    nloss, prec1 = loss(x, y)
    print(nloss, prec1)
