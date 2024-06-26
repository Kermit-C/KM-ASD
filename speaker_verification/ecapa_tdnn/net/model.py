'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)

class FbankAug(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
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

        # mask_len每个batch要mask不同的长度
        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        # mask_pos每个batch要mask起始的位置
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        # 把要mask的变为1其他的为0
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
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

class ECAPA_TDNN1(nn.Module):

    def __init__(self, C, n_mels=80, imput="fbank"):  # 中间的变化维度1024

        super(ECAPA_TDNN1, self).__init__()

        self.imput = imput
        if imput == "MFCC":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mels),
            )
        elif imput == "Spectrogram":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=1023, win_length=400, hop_length=160, window_fn=torch.hamming_window),
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=n_mels),
                )

        self.specaug = FbankAug()  # Spec augmentation
        if imput == "Spectrogram":
            n_mels = 512
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            if self.imput == "fbank":
                x = x.log()
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.instancenorm(x)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN1_xs(nn.Module):

    def __init__(self, C, n_mels=80, imput="fbank"):  # 中间的变化维度1024

        super(ECAPA_TDNN1_xs, self).__init__()

        self.imput = imput
        if imput == "MFCC":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mels),
            )
        elif imput == "Spectrogram":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=1023, win_length=400, hop_length=160, window_fn=torch.hamming_window),
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=n_mels),
                )

        self.specaug = FbankAug()  # Spec augmentation
        if imput == "Spectrogram":
            n_mels = 512
        self.instancenorm = nn.InstanceNorm1d(n_mels)
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.fc4 = nn.Conv1d(1536, 192, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            if self.imput == "fbank":
                x = x.log()
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.instancenorm(x)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]
        c = x.size()[1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        x_n = self.fc4(torch.mean((x * w).reshape(-1, c, 10, t//10), 3))
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x, x_n

class ECAPA_TDNN_flow(nn.Module):

    def __init__(self, C, n_mels=80, imput="fbank"):  # 中间的变化维度1024

        super(ECAPA_TDNN_flow, self).__init__()

        self.imput = imput
        if imput == "MFCC":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mels),
            )
        elif imput == "Spectrogram":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=1023, win_length=400, hop_length=160, window_fn=torch.hamming_window),
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=n_mels),
                )

        self.specaug = FbankAug()  # Spec augmentation
        if imput == "Spectrogram":
            n_mels = 512
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        # x1 = torch.roll(x, 1, 1)
        # x = x - x1
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x1 = torch.roll(x, 1, 2)
            x = x - x1 + 1e-6
            # if self.imput == "fbank":
            #     x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

# 统计池化层对x直接自注意力不拼接均值方差
class ECAPA_TDNN2(nn.Module):

    def __init__(self, C):  # 中间的变化维度1024

        super(ECAPA_TDNN2, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )

        self.specaug = FbankAug()  # Spec augmentation

        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(1538, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        # global_x = x + torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t) + torch.sqrt(torch.var(x, dim=2, keepdim=True)c).repeat(1, 1, t)
        x = torch.cat((x, torch.mean(x, dim=2, keepdim=True), torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)), dim=2)
        global_x = torch.cat((x, torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1, keepdim=True).clamp(min=1e-4)), dim=1)
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN_RNN(nn.Module):

    def __init__(self, C, n_mels=80, imput="fbank"):  # 中间的变化维度1024

        super(ECAPA_TDNN_RNN, self).__init__()

        self.imput = imput
        if imput == "MFCC":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mels),
            )
        elif imput == "Spectrogram":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=1023, win_length=400, hop_length=160, window_fn=torch.hamming_window),
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=n_mels),
                )

        self.specaug = FbankAug()  # Spec augmentation
        if imput == "Spectrogram":
            n_mels = 512
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.rnn = nn.RNN(1536, 768, 2, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        # x = x.reshape(-1, x.size()[-1])  #lrfinder时用
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            if self.imput == "fbank":
                x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x).transpose(1, 2)
        output, h = self.rnn(x)
        x = h.permute(1, 0, 2).flatten(1)

        # t = x.size()[-1]
        #
        # global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        #
        # w = self.attention(global_x)
        #
        # mu = torch.sum(x * w, dim=2)
        # sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        #
        # x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN_wav(nn.Module):

    def __init__(self, C):  # 中间的变化维度1024

        super(ECAPA_TDNN_wav, self).__init__()

        # self.torchfbank = torch.nn.Sequential(
        #     PreEmphasis(),
        #     torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
        #                                          f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        # )

        self.specaug = FbankAug()  # Spec augmentation

        self.conv1 = nn.Conv1d(160, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(1536, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=True):
        with torch.no_grad():
        #     x = self.torchfbank(x) + 1e-6
        #     x = x.log()
        #     x = x - torch.mean(x, dim=-1, keepdim=True)
            x = x.view(-1, 160, 300)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = x + torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t) + torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN1_BN(nn.Module):

    def __init__(self, C, n_mels=80, imput="fbank"):  # 中间的变化维度1024

        super(ECAPA_TDNN1_BN, self).__init__()

        self.imput = imput
        if imput == "MFCC":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mels),
            )
        elif imput == "Spectrogram":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=1023, win_length=400, hop_length=160, window_fn=torch.hamming_window),
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=n_mels),
                )

        self.specaug = FbankAug()  # Spec augmentation
        if imput == "Spectrogram":
            n_mels = 512
        self.norm = nn.BatchNorm1d(n_mels)
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            if self.imput == "fbank":
                x = x.log()
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.norm(x)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN1_LN(nn.Module):

    def __init__(self, C, n_mels=80, imput="fbank"):  # 中间的变化维度1024

        super(ECAPA_TDNN1_LN, self).__init__()

        self.imput = imput
        if imput == "MFCC":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mels),
            )
        elif imput == "Spectrogram":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=1023, win_length=400, hop_length=160, window_fn=torch.hamming_window),
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=n_mels),
                )

        self.specaug = FbankAug()  # Spec augmentation
        if imput == "Spectrogram":
            n_mels = 512
        self.norm = nn.LayerNorm([n_mels, 300])
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            if self.imput == "fbank":
                x = x.log()
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.norm(x)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

class ECAPA_TDNN1_CN(nn.Module):

    def __init__(self, C, n_mels=80, imput="fbank"):  # 中间的变化维度1024

        super(ECAPA_TDNN1_CN, self).__init__()

        self.imput = imput
        if imput == "MFCC":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mels),
            )
        elif imput == "Spectrogram":
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.Spectrogram(n_fft=1023, win_length=400, hop_length=160, window_fn=torch.hamming_window),
            )
        else:
            self.torchfbank = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                                                     f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=n_mels),
                )

        self.specaug = FbankAug()  # Spec augmentation
        if imput == "Spectrogram":
            n_mels = 512
        self.norm = nn.InstanceNorm1d(300)
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)


    def forward(self, x, aug=True):
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            if self.imput == "fbank":
                x = x.log()
            # x = x - torch.mean(x, dim=-1, keepdim=True)
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim  32240 => 202, 35760=>224 ,47920=>300
    # x = torch.zeros(32, 48000).cuda()
    # model = ECAPA_TDNN_wav(1024).cuda()
    x = torch.randn(10, 47920).cuda()
    model = ECAPA_TDNN1_xs(1024).cuda()
    # print(model)
    # summary(model, input_size=(tuple(x.shape)))
    out = model(x)
    print(out[0].shape, out[1].shape)