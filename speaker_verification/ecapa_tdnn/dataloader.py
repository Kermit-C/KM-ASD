import glob
import os
import random

import numpy as np
import soundfile
import torch
from scipy import signal
from torch.utils.data import Dataset


def round_down(num, divisor):
    return num - (num % divisor)

# (1, 47920)
def loadWAV(filename, max_frames, segment=1, evalmode=False, num_eval=10):

    # Maximun audio length
    max_audio = (max_frames - 2) * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    audio = np.array(audio)
    # print(audio.shape)
    # print(audio)
    # print(sample_rate)
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
        audio = audio[::3]
        # print(audio.shape)
    audiosize = audio.shape[0]
    if audiosize <= max_audio:  # 给不足max_frams的padding
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')  # 0是前面的padding，后面padding shortage本身的循环
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)  # 测试的话分为10段
    else:
        startframe = np.array([np.int64(random.random() * (audiosize-max_audio)) for _ in range(segment)])
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(np.float64)

    return feat


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames, segment):

        self.max_frames = max_frames
        self.max_audio = (max_frames - 2) * 160 + 240
        self.segment = segment

        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, segment=self.segment, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, fs = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float64), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))

        return signal.convolve(audio, rir, mode='full')[:, :self.max_audio]


class train_dataset_loader(Dataset):
    def __init__(self, train_list, train_path, augment, musan_path, rir_path, max_frames, nPerSpeaker):
        self.train_list = train_list
        self.max_frames = max_frames
        self.augment = augment
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.nPerSpeaker = nPerSpeaker
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames, segment=nPerSpeaker)

        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        np.random.shuffle(lines)

        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()
            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path, data[1])
            self.data_list.append(filename)
            self.data_label.append(speaker_label)

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False, segment=self.nPerSpeaker)
        if self.augment:
            augtype = random.randint(0, 4)  # 包括0，4
            if augtype == 1:
                audio = self.augment_wav.reverberate(audio)
            elif augtype == 2:
                audio = self.augment_wav.additive_noise('music', audio)
            elif augtype == 3:
                audio = self.augment_wav.additive_noise('speech', audio)
            elif augtype == 4:
                audio = self.augment_wav.additive_noise('noise', audio)

        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames, num_eval):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_path = test_path
        self.test_list = test_list

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path, self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


def built_train_loader(cfg):
    if cfg.DATA.LOADER == "dataloader":
        train_dataset = train_dataset_loader(train_list=cfg.DATA.TRAIN_LIST, train_path=cfg.DATA.TRAIN_PATH,
                                             augment=cfg.DATA.AUGMENT, musan_path=cfg.DATA.MUSAN_PATH,
                                             rir_path=cfg.DATA.RIR_PATH, max_frames=cfg.DATA.MAX_FRAMES,
                                             nPerSpeaker=cfg.nPerSpeaker)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.DATA.BATCH_SIZE,
            num_workers=cfg.DATA.NUM_WORKERS,
            pin_memory=cfg.DATA.PIN_MEMORY,
            drop_last=True,
        )
    print("dataloader:", cfg.DATA.LOADER)
    return train_loader


if __name__ == "__main__":
    # audio = loadWAV("data/voxceleb1/id10969/qSSfnPaIY70/00001.wav", max_frames=300, segment=1, evalmode=False, num_eval=10)
    # print(audio.shape)
    train_dataset = train_dataset_loader(train_list="data/train_list_vox.txt", augment=True,
                                 musan_path="data/musan_split", rir_path="data/RIRS_NOISES/simulated_rirs",
                                 max_frames=400, train_path="data/voxceleb", nPerSpeaker=1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        pin_memory=False,
        drop_last=True,
        num_workers=16,
    )
    x, y = iter(train_loader).next()
    print("x:", x.shape, "y:", y.shape)  #[32, 1, 47920]
