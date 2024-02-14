import itertools
import os
import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from speaker_verification.ecapa_tdnn.utils.log import init_log

from .dataloader import test_dataset_loader
from .dataloader_wav import test_dataset_wav_loader
from .tuneThreshold import *


class SpeakerNet(nn.Module):

    def __init__(self, cfg, model, loss):
        super(SpeakerNet, self).__init__()

        self.cfg = cfg
        self.model = model
        self.loss = loss
        self.nPerSpeaker = cfg.nPerSpeaker

    def forward(self, data, label=None):

        if self.cfg.TRAIN.MIXUP and label != None:
            alpha = self.cfg.TRAIN.ALPHA
            lam = numpy.random.beta(alpha, alpha)
            index = torch.randperm(data.size(0))
            data = lam * data + (1 - lam) * data[index, :]
            label_a, label_b = label, label[index]
            data = data.reshape(-1, data.size()[-1])
            outp = self.model(data)
            emb = outp.reshape(-1, self.nPerSpeaker, outp.size()[-1]).squeeze(1)
            nloss_a, prec1_a = self.loss(emb, label_a)
            nloss_b, prec1_b = self.loss(emb, label_b)
            nloss, prec1 = lam * nloss_a + (1 - lam) * nloss_b, lam * prec1_a + (1 - lam) * prec1_b
        elif self.cfg.TRAIN.ADDUP and label != None:
            alpha = self.cfg.TRAIN.ALPHA
            # lam = numpy.random.beta(alpha, alpha)
            lam = 1
            data = data[:, 0, :] + data[:, 1, :]
            data = data.reshape(-1, data.size()[-1])
            emb = self.model(data)
            nloss, prec1 = self.loss(emb, label)
        # elif self.cfg.TRAIN.ADDUP and label == None:
        #     lam = 1
        #     data = lam * data[0:5, :] + lam * data[5:, :]
        #     data = data.reshape(-1, data.size()[-1])
        #     outp = self.model(data)
        #     return outp
        elif self.cfg.LOSS.NAME == "AamSoftmax_XS":
            data = data.reshape(-1, data.size()[-1])
            outp = self.model(data)

            if label == None:
                return outp[0]

            else:
                emb = outp
                nloss, prec1 = self.loss(emb, label)
        else:
            data = data.reshape(-1, data.size()[-1])
            outp = self.model(data)

            if label == None:
                return outp

            else:
                emb = outp.reshape(-1, self.nPerSpeaker, outp.size()[-1]).squeeze(1)
                nloss, prec1 = self.loss(emb, label)

        return nloss, prec1


class Trainer(object):

    def __init__(self, cfg, model, optimizer, scheduler, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        logging = init_log(cfg.SAVE_DIR)
        self._print = logging.info
        self.best = 0
        self.test_eer = 0
        self.test_mindcf = 0
        self.best_model = []

    def train(self, epoch, dataloader):
        self.model.train()
        pbar = tqdm(dataloader)
        loss = 0
        top1 = 0
        index = 0
        counter = 0

        for data in pbar:
            # x:[32, 1, 47920] label:[32]
            x, label = data[0].to(self.device), data[1].long().to(self.device)
            nloss, prec1 = self.model(x, label)
            # print(nloss)

            self.optimizer.zero_grad()
            nloss.backward()
            self.optimizer.step()
            if self.cfg.OPTIMIZER.SCHEDULER == "CyclicLR":
                self.scheduler.step()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            index += x.size(0)
            counter += 1

            if self.cfg.WANDB:
                wandb.log({
                    "epoch": epoch,
                    "train_acc": top1 / counter,
                    "train_loss": loss / counter,
                })
            pbar.set_description("Train Epoch:%3d ,loss:%.3f, acc:%.3f" % (epoch, loss / counter, top1 / counter))

        lr = self.optimizer.param_groups[0]['lr']
        self._print('epoch:{} - lr: {} train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
            epoch, lr, loss / counter, top1 / counter, index))
        if self.cfg.OPTIMIZER.SCHEDULER == "StepLR":
            self.scheduler.step()

    def test(self, epoch, test_list, test_path, eval_frames, num_eval=10):

        self.model.eval()
        feats = {}

        # read all lines
        with open(test_list) as f:
            lines = f.readlines()
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        # Define test data loader
        if self.cfg.DATA.LOADER == "dataloader_wav":
            test_dataset = test_dataset_wav_loader(setfiles, test_path, eval_frames=eval_frames, num_eval=num_eval)
        else:
            test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=eval_frames, num_eval=num_eval)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            drop_last=False,
            sampler=None,
            pin_memory=self.cfg.DATA.PIN_MEMORY,
        )

        # Extract features for every wav
        for idx, data in enumerate(tqdm(test_loader)):

            inp1 = data[0][0].to(self.device)  # (data[0]:(1,10,1024),data[1]:['id10270/GWXujl-xAVM/00017.wav',])
            with torch.no_grad():
                ref_feat = self.model(inp1).detach().cpu()

            feats[data[1][0]] = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        # Read files and compute all scores
        for idx, line in enumerate(tqdm(lines)):

            data = line.split()

            # Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data
            ref_feat = feats[data[1]].to(self.device)
            com_feat = feats[data[2]].to(self.device)

            if self.model.loss.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # dist = F.pairwise_distance(ref_feat.unsqueeze(-1),
            #                            com_feat.unsqueeze(-1).transpose(0, 2)).detach().cpu().numpy()
            #
            # score = -1 * numpy.mean(dist)
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1),
                                       com_feat.unsqueeze(-1).transpose(0, 2)).detach().cpu().numpy()
            score = numpy.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

        result = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])
        fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, self.cfg.EVALUATION.DCF_P_TARGET, self.cfg.EVALUATION.DCF_C_MISS, self.cfg.EVALUATION.DCF_C_FA)
        self.test_eer = result[1]
        self.test_mindcf = mindcf
        self.threshold = threshold
        if self.cfg.WANDB:
            wandb.log({
                "test_eer": self.test_eer,
                "test_MinDCF": self.test_mindcf,
            })
        self._print('epoch:{} - test EER: {:.3f} and test MinDCF: {:.3f} total sample: {} threshold: {:.3f}'.format(
            epoch, self.test_eer, self.test_mindcf, len(lines), self.threshold))

        return self.test_eer

    def save_model(self, epoch):
        if self.test_eer < self.best or self.best == 0:
            self.best = self.test_eer
            if self.cfg.WANDB:
                wandb.run.summary["best_accuracy"] = self.best
            model_state_dict = self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
            scheduler_state_dict = self.scheduler.state_dict()
            file_save_path = 'epoch:%d,EER:%.4f,MinDCF:%.4f' % (epoch, self.test_eer, self.test_mindcf)
            if not os.path.exists(self.cfg.SAVE_DIR):
                os.mkdir(self.cfg.SAVE_DIR)
            torch.save({
                'epoch': epoch,
                'test_eer':  self.test_eer,
                'test_mindcf': self.test_mindcf,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'scheduler_state_dict': scheduler_state_dict},
                os.path.join(self.cfg.SAVE_DIR, file_save_path))
            self.best_model.append(file_save_path)
            if len(self.best_model) > 3:
                del_file = os.path.join(self.cfg.SAVE_DIR, self.best_model.pop(0))
                if os.path.exists(del_file):
                    os.remove(del_file)
                else:
                    print("no exists {}".format(del_file))
        # 每20个epoch保存一下
        if epoch % 20 == 0:
            model_state_dict = self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
            scheduler_state_dict = self.scheduler.state_dict()
            file_save_path = 'epoch:%d,EER:%.4f,MinDCF:%.4f' % (epoch, self.test_eer, self.test_mindcf)
            if not os.path.exists(os.path.join(self.cfg.SAVE_DIR, file_save_path)):
                torch.save({
                    'epoch': epoch,
                    'test_eer':  self.test_eer,
                    'test_mindcf': self.test_mindcf,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_state_dict,
                    'scheduler_state_dict': scheduler_state_dict},
                    os.path.join(self.cfg.SAVE_DIR, file_save_path))
