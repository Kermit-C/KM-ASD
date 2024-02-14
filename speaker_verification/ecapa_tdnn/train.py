import argparse
import os
import wandb
import torch.backends.cudnn as cudnn
import torch
from torch import optim
from config import get_config
import numpy as np
from model import built_model
from loss import built_loss
from optimizer import build_optimizer
from scheduler import build_scheduler
from dataloader import built_train_loader
from dataloader_wav import built_train_wav_loader
from SpeakerNet import *
from torchinfo import summary


def get_cfg():
    parser = argparse.ArgumentParser(description="vox verification")
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch_size', default=None, type=int, help="batch size")
    parser.add_argument("--resume", default=None, type=str, help="resume path")
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Eval only')
    parser.add_argument('--eval_model', default=None, type=str, help="eval model path")
    parser.add_argument("--wandb", action='store_true', default=False, help='use wandb to log ')
    parser.add_argument("--note", type=str, default="", help='wandb note')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main():

    # --------参数解析--------
    cfg, args = get_cfg()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    save_yaml_path = os.path.join(cfg.SAVE_DIR, "config.yaml")
    with open(save_yaml_path, "w") as f:
        f.write(cfg.dump())

    if cfg.WANDB:
        wandb.login(host="http://49.233.11.7:8080", key="local-7dc64cc63778f0723dc202d2624a97cef7043120")
        wandb.init(project=cfg.PROJECT, name=cfg.NAME, config=cfg, save_code=True, notes=args.note)

    # setting seed
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # --------数据加载--------
    if cfg.DATA.LOADER == "dataloader_wav":
        train_loader = built_train_wav_loader(cfg)
    else:
        train_loader = built_train_loader(cfg)
    x, y = iter(train_loader).next()
    print('x.shape:', x.shape, 'y.shape:', y.shape)  # [128, 1, 47920]
    print('x.dtype:', x.dtype, 'y.dtype:', y.dtype)  # [128]

    # --------模型加载---------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = built_model(cfg).to(device)
    loss = built_loss(cfg).to(device)
    model = SpeakerNet(cfg, model=model, loss=loss)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    if cfg.TRAIN.MIXUP:
        print("mix up , alpha:", cfg.TRAIN.ALPHA)
    if cfg.TRAIN.ADDUP:
        print("add up , alpha:", cfg.TRAIN.ALPHA)
    summary(model, input_size=(tuple(x.shape)))

    if cfg.MODEL.RESUME:
        # ckpt = torch.load(cfg.MODEL.RESUME, map_location="cpu")
        ckpt = torch.load(cfg.MODEL.RESUME)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # start_epoch = ckpt['epoch'] + 1
        start_epoch = 0
        print("checkpoint加载完毕!")
    else:
        start_epoch = 1

    # ---------训练----------
    trainer = Trainer(cfg, model, optimizer, scheduler, device)
    # 测试
    if cfg.EVAL:
        trainer.test(0, cfg.DATA.TEST_LIST, cfg.DATA.TEST_PATH, cfg.DATA.EVAL_FRAMES)
    # 训练
    else:
        it = 0
        base_lr = cfg.OPTIMIZER.BASE_LR
        max_lr = cfg.OPTIMIZER.MAX_LR
        min_eer = float("inf")
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
            trainer.train(epoch, train_loader)
            if epoch % cfg.TRAIN.TEST_INTERVAL == 0:
                eer = trainer.test(epoch, cfg.DATA.TEST_LIST, cfg.DATA.TEST_PATH,
                                   cfg.DATA.EVAL_FRAMES)
                # ----Cycling LR-----
                # if cfg.OPTIMIZER.SCHEDULER == "CyclicLR":
                #     if eer < min_eer:
                #         min_eer = eer
                #         it = 0
                #
                #     else:
                #         it += 1
                #
                #         if it >= 8:
                #             base_lr = base_lr * 0.1
                #             max_lr = max_lr * 0.1
                #             trainer.scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                #                                                             mode='triangular2',
                #                                                             step_size_up=6000, cycle_momentum=False)
                #             it = 0
                # ------Cycling LR-----
                if cfg.OPTIMIZER.SCHEDULER == "ReduceLROnPlateau":
                    scheduler.step(eer)
                trainer.save_model(epoch)

    print("finishing")


if __name__ == "__main__":
    main()
















