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

class AamSoftmax_XS(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, **kwargs):
        super(AamSoftmax_XS, self).__init__()
        self.test_normalize = True
        self.aamsoftmax = AamSoftmax(nOut, nClasses, margin=margin, scale=scale, easy_margin=easy_margin)
        self.s = 30

    def forward(self, x, label=None):
        loss1, prec1 = self.aamsoftmax(x[0], label)
        emb = x[0] / torch.norm(x[0], dim=1, keepdim=True)
        x_emb = x[1].permute(0, 2, 1)
        x_emb = x_emb / torch.norm(x_emb, dim=2, keepdim=True)
        loss2 = torch.mean(torch.sum(torch.abs(x_emb @ emb.unsqueeze(-1)), 1))
        loss = loss1 + self.s * loss2
        # print("loss1:", loss1, "loss2:", loss2, "loss:", loss)

        return loss, prec1


if __name__ == "__main__":
    x = torch.randn(32, 192), torch.randn(32, 192, 300)
    y = torch.randint(1000, size=(32,))
    print(y.shape)
    loss = AamSoftmax_XS(192, 1000)
    nloss, prec1 = loss(x, y)
    print(nloss, prec1)
