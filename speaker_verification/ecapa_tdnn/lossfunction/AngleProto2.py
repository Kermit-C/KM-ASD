import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.ecapa_tdnn.utils.acc import accuracy


class AngleProto2(nn.Module):

    def __init__(self, init_w=10.0, init_b=5.0):
        super(AngleProto2, self).__init__()

        self.test_normalize = True

        # self.w = nn.Parameter(torch.tensor(init_w))
        # self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised AngleProto2')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]

        stepsize = out_anchor.size()[0]
        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))
        # torch.clamp(self.w, 1e-6)
        cos_sim_matrix = (1 - cos_sim_matrix)
        # cos_sim_matrix = (1 - cos_sim_matrix)

        label_matrix = label.unsqueeze(0)-label.unsqueeze(1)
        positive_mask = torch.eq(label_matrix, 0.0).float()
        negative_mask = 1 - positive_mask
        nloss = torch.sum(cos_sim_matrix * positive_mask) / torch.sum(positive_mask) - torch.sum(cos_sim_matrix * negative_mask) / torch.sum(negative_mask) + 2
        # nloss = torch.sum(cos_sim_matrix * positive_mask) / torch.sum(positive_mask)
        # print(torch.sum(cos_sim_matrix * positive_mask) / torch.sum(positive_mask), torch.sum(cos_sim_matrix * negative_mask) / torch.sum(negative_mask))
        prec1 = accuracy(-cos_sim_matrix.detach(), torch.tensor(range(0, len(label))).cuda(), topk=(1,))[0]


        return nloss, prec1


if __name__ == "__main__":
    x = torch.randn(32, 2, 512).cuda()
    y = torch.randint(1000, size=(32,)).cuda()
    print(x.shape, y.shape)
    loss = AngleProto2()
    nloss, prec1 = loss(x, y)
    print(nloss, prec1)
