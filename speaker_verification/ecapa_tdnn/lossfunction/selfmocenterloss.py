import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.ecapa_tdnn.utils.acc import accuracy


class SAMomentumCenter(nn.Module):
    def __init__(self, nOut, nClasses):
        super(SAMomentumCenter, self).__init__()
        self.test_normalize = True
        self.in_feats = nOut
        self.nClasses = nClasses
        self.centers = torch.randn(nClasses, nOut).cuda()
        self.ce = nn.CrossEntropyLoss()
        self.fc = nn.Linear(nOut, 1)
        self.classhead = nn.Linear(nOut, nClasses)

    def forward(self, x, label=None):
        label_idex = []
        # 叠加的x
        # requires_grad
        cneter_add = torch.zeros(self.nClasses, self.in_feats,requires_grad=True).cuda()
        for i, l in enumerate(label):
            if l not in label_idex:
                label_idex.append(l)
                cneter_add[l] = cneter_add[l] + x[i]
            else:
                cneter_add[l] = (cneter_add[l] + x[i]) / 2
        # 调节中心和x叠加的参数

        alpha = torch.sigmoid(self.fc(self.centers)).squeeze(1)

        label_idex = torch.tensor(label_idex)
        label_add = torch.zeros(self.nClasses).index_fill_(0, label_idex, 1).cuda()
        center_mask = 1 - label_add
        # with torch.no_grad():

        new_center = self.centers * (alpha * label_add + center_mask).unsqueeze(-1) + cneter_add * ((1-alpha) * label_add).unsqueeze(-1)
        self.centers = new_center.detach()
        center_label = torch.from_numpy(numpy.asarray(range(0, self.nClasses))).cuda()
        lossC = self.ce(self.classhead(new_center), center_label)

        output = self.classhead(x)
        nloss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        loss = nloss + lossC
        # print(nloss, lossC)
        # prec1 = torch.tensor(0)
        return loss, prec1


if __name__ == "__main__":
    x = torch.randn(32, 512).cuda()
    y = torch.randint(1000, size=(32,)).cuda()
    print(x.shape, y.shape)
    loss = SAMomentumCenter(512, 1000).cuda()
    nloss, prec1 = loss(x, y)
    print(nloss, prec1)
