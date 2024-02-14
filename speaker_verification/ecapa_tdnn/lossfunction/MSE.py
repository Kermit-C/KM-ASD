import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from speaker_verification.ecapa_tdnn.utils.acc import accuracy


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()

        self.test_normalize = True
        self.mse = torch.nn.MSELoss()

        print('Initialised MSE')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]

        # print(label)
        nloss = self.mse(out_positive, out_anchor)
        # nloss = self.criterion(cos_sim_matrix, label)
        # print("lossC:", self.criterion(cos_sim_matrix, label), "lossM:", self.mse(out_positive, out_anchor))
        prec1 = torch.tensor(0)

        return nloss, prec1
