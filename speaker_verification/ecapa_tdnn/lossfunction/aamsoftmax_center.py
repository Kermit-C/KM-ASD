import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.ecapa_tdnn.utils.acc import accuracy


class AamSoftmaxCenterW(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, lam=0.1, **kwargs):
        super(AamSoftmaxCenterW, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.nClasses = nClasses
        # lam 调节softmaxloss和centerloss比例参数
        self.lam = lam
        # w是类的中心
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut).cuda(), requires_grad=True)
        # centers中心
        # self.centers = nn.Parameter(torch.randn(self.nClasses, self.in_feats).cuda())
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

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

        # center loss
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.nClasses) + \
                  torch.pow(self.weight, 2).sum(dim=1, keepdim=True).expand(self.nClasses, batch_size).t()
        distmat.addmm(x, self.weight.t(), beta=1, alpha=-2)

        classes = torch.arange(self.nClasses).long().cuda()
        labels = label.unsqueeze(1).expand(batch_size, self.nClasses)
        mask = labels.eq(classes.expand(batch_size, self.nClasses))

        dist = distmat * mask.float()
        lossC = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # print("lossS", lossS, "lossC", lossC)

        loss = lossS + self.lam * lossC
        return loss, prec1
        # return loss

class AamSoftmaxCenterWA(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, lam=10, **kwargs):
        super(AamSoftmaxCenterWA, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.nClasses = nClasses
        # lam 调节softmaxloss和centerloss比例参数
        self.lam = lam
        # w是类的中心
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut).cuda(), requires_grad=True)
        # centers中心
        # self.centers = nn.Parameter(torch.randn(self.nClasses, self.in_feats).cuda())
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmaxCenterA margin %.3f scale %.3f lam %.3f' % (self.m, self.s, self.lam))

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

        # center angle loss
        batch_size = x.size(0)
        cos_distmat = F.cosine_similarity(x.unsqueeze(-1), self.weight.unsqueeze(-1).transpose(0, 2))

        classes = torch.arange(self.nClasses).long().cuda()
        labels = label.unsqueeze(1).expand(batch_size, self.nClasses)
        mask = labels.eq(classes.expand(batch_size, self.nClasses))

        dist = self.s * cos_distmat * mask.float()
        lossC = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # print("lossS", lossS, "lossC", lossC)

        loss = lossS + self.lam * lossC
        return loss, prec1


class AamSoftmaxCenter(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, lam=0.1, **kwargs):
        super(AamSoftmaxCenter, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.nClasses = nClasses
        # lam 调节softmaxloss和centerloss比例参数
        self.lam = lam
        # w是类的中心
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut).cuda(), requires_grad=True)
        # centers中心
        self.centers = nn.Parameter(torch.randn(self.nClasses, self.in_feats).cuda())
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

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

        # center loss
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.nClasses) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.nClasses, batch_size).t()
        distmat.addmm(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.nClasses).long().cuda()
        labels = label.unsqueeze(1).expand(batch_size, self.nClasses)
        mask = labels.eq(classes.expand(batch_size, self.nClasses))

        dist = distmat * mask.float()
        lossC = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # print("lossS", lossS, "lossC", lossC)

        loss = lossS + self.lam * lossC
        return loss, prec1
        # return loss

class AamSoftmaxCenterA(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, lam=100, **kwargs):
        super(AamSoftmaxCenterA, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.nClasses = nClasses
        # lam 调节softmaxloss和centerloss比例参数
        self.lam = lam
        # w是类的中心
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut).cuda(), requires_grad=True)
        # centers中心
        self.centers = nn.Parameter(torch.randn(self.nClasses, self.in_feats).cuda())
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmaxCenterA margin %.3f scale %.3f lam %.3f' % (self.m, self.s, self.lam))

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

        # center angle loss
        batch_size = x.size(0)
        cos_distmat = F.cosine_similarity(x.unsqueeze(-1), self.centers.unsqueeze(-1).transpose(0, 2))

        classes = torch.arange(self.nClasses).long().cuda()
        labels = label.unsqueeze(1).expand(batch_size, self.nClasses)
        mask = labels.eq(classes.expand(batch_size, self.nClasses))

        dist = self.s * cos_distmat * mask.float()
        lossC = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        # print("lossS", lossS, "lossC", lossC)

        loss = lossS + self.lam * lossC
        return loss, prec1


if __name__ == "__main__":
    x = torch.randn(32, 512).cuda()
    y = torch.randint(1000, size=(32,)).cuda()
    print(x.shape, y.shape)
    loss = AamSoftmaxCenterA(512, 1000)
    nloss, prec1 = loss(x, y)
    print(nloss, prec1)
