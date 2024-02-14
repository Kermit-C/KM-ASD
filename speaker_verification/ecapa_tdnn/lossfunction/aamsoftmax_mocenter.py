import math

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.ecapa_tdnn.utils.acc import accuracy


class AamSoftmaxMoCenter(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, lam=0.1, **kwargs):
        super(AamSoftmaxMoCenter, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.nClasses = nClasses
        # lam 调节softmaxloss和centerloss比例参数
        self.lam = lam
        # w是类的中心
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut).cuda(), requires_grad=True)

        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        # momentum centers中心
        self.centers = torch.zeros(nClasses, nOut)
        self.fc = nn.Linear(nOut, nOut)
        print('Initialised AAMSoftmaxCenter margin %.3f scale %.3f lam %.3f' % (self.m, self.s, self.lam))

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
        lossS = self.ce(output, label)

        # momuum center loss
        label_idex = []
        # 叠加的x
        cneter_add = torch.zeros(self.nClasses, self.in_feats)
        for i, l in enumerate(label):
            if l not in label_idex:
                label_idex.append(l)
                cneter_add[l] = cneter_add[l] + x[i]
            else:
                cneter_add[l] = torch.mean(cneter_add[l], x[i])
        # 调节中心和x叠加的参数
        alpha = F.sigmoid(self.fc(self.centers))
        label_add = torch.zeros(self.nClasses).index_fill_(0, label_idex, 1)
        center_mask = 1 - label_add
        self.centers = self.centers * (alpha * label_add + center_mask) + cneter_add * (1 - alpha) * center_mask
        center_label = torch.from_numpy(numpy.asarray(range(0, self.nClasses))).cuda()
        lossC = self.ce(self.centers, center_label)

        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # print("lossS", lossS, "lossC", lossC)

        loss = lossS + self.lam * lossC
        return loss, prec1
        # return loss


if __name__ == "__main__":
    x = torch.randn(32, 512).cuda()
    y = torch.randint(1000, size=(32,)).cuda()
    print(x.shape, y.shape)
    loss = AamSoftmaxMoCenter(512, 1000)
    nloss, prec1 = loss(x, y)
    print(nloss, prec1)
