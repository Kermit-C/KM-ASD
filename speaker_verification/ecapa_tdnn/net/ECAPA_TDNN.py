import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchinfo import summary

''' Res2Conv1d + BatchNorm1d + ReLU
'''
class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out



''' Conv1d + BatchNorm1d + ReLU
'''
class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))



''' The SE connection of 1D case.
'''
class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out



''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN.yaml model, not here.
'''

class SE_Res2Block(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, scale):
         super().__init__()
         self.block = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
            Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
            SE_Connect(channels)
         )

    def forward(self, x):
        out = self.block(x)
        return out + x



''' Attentive weighted mean and standard deviation pooling.
'''
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1) # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1) # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)



''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".
    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''
class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, embd_dim=192):
        super().__init__()

        self.torchfb = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                                 hop_length=160, f_min=20, f_max=7600, pad=0, n_mels=80,
                                                 window_fn=torch.hamming_window)
        )
        self.specaug = FbankAug()
        self.instancenorm = nn.InstanceNorm1d(80)
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)

        # self.pooling = AttentiveStatsPool(cat_channels, 128)
        # 修改pooling
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn1 = nn.BatchNorm1d(1536 * 2)
        self.linear = nn.Linear(1536 * 2, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfb(x) + 1e-61
            x = x.log()
            x = self.instancenorm(x)
            if aug == True:
                x = self.specaug(x)
        # print(x.shape)
        # x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        # print(out.shape)

        # 修改---pooling-------
        t = out.size()[-1]
        global_x = torch.cat((out, torch.mean(out, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(out, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)

        mu = torch.sum(out * w, dim=2)
        sg = torch.sqrt((torch.sum((out ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        out = torch.cat((mu, sg), 1)
        out = self.bn1(out)
        # out = self.bn1(self.pooling(out))
        # -------pooling----
        # print(out.shape)
        out = self.bn2(self.linear(out))
        return out

class ECAPA_TDNN_pool(nn.Module):
    def __init__(self, in_channels=80, channels=512, embd_dim=192):
        super().__init__()

        self.torchfb = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                                 hop_length=160, f_min=20, f_max=7600, pad=0, n_mels=80,
                                                 window_fn=torch.hamming_window)
        )
        self.specaug = FbankAug()
        self.instancenorm = nn.InstanceNorm1d(80)  # 这个地方不一样，做了以归一化
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)

        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(cat_channels * 2)
        self.linear = nn.Linear(cat_channels * 2, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfb(x) + 1e-6
            x = x.log()
            x = self.instancenorm(x)
            if aug == True:
                x = self.specaug(x)
        # print(x.shape)
        # x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        # print(out.shape)
        out = torch.cat((out, torch.mean(out, dim=2, keepdim=True),
                   torch.sqrt(torch.var(out, dim=2, keepdim=True).clamp(min=1e-4))), dim=2)
        out = self.bn1(self.pooling(out))
        # -------pooling----
        out = self.bn2(self.linear(out))
        return out
class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        if input.size() != 2:
            input = input.reshape(-1, input.size()[-1])
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        # mask_len每个batch要mask不同的长度 mask_len(32,1,1)
        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        # mask_pos每个batch要mask起始的位置 (32,1,1)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        # 把要mask的变为1其他的为0
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        # 把mask_pos到mask_pos + mask_len变为1（32，1，200））
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        # 压缩dim=1纬度（b，200）
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim  32240 => 202, 35760=>224 ,47920=>300
    x = torch.zeros(32, 35760).cuda()
    model = ECAPA_TDNN_pool(in_channels=80, channels=1024, embd_dim=192).cuda()
    # print(model)
    summary(model, input_size=(tuple(x.shape)))
    out = model(x)
    print(out.shape)    # should be [2, 192]

