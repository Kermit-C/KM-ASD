import os
import yaml
#import yacs
from yacs.config import CfgNode as CN
from datetime import datetime


_C = CN()

# Base config files
_C.BASE = ['']

# project name
_C.PROJECT = "speaker_verification"
_C.NAME = "ECAPA_TDNN"
_C.SAVE_DIR = "train_models"
_C.SEED = 100
_C.nPerSpeaker = 1
_C.DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Train settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCH = 300
_C.TRAIN.TEST_INTERVAL = 1
_C.TRAIN.MIXUP = False
_C.TRAIN.ALPHA = 1.0  # mixup alphe
_C.TRAIN.ADDUP = False

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# dataset path
_C.DATA.TRAIN_LIST = "data/train_list.txt"
_C.DATA.TEST_LIST = "data/veri_list.txt"
_C.DATA.TRAIN_PATH = "data/voxceleb2"
_C.DATA.TEST_PATH = "data/voxceleb1"
_C.DATA.MUSAN_PATH = "data/musan_split"  # 噪声文件
_C.DATA.RIR_PATH = "data/RIRS_NOISES/simulated_rirs"  # 混响文件

_C.DATA.LOADER = "dataloader"
_C.DATA.BATCH_SIZE = 32
_C.DATA.MAX_FRAMES = 300  # 训练时帧长300
_C.DATA.EVAL_FRAMES = 300  # 测试帧长
_C.DATA.NUM_WORKERS = 16  # Number of data loading threads
_C.DATA.AUGMENT = True  # 数据增强
_C.DATA.FBANK_AUG = True  # 数据增强遮挡
_C.DATA.PIN_MEMORY = False

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "ECAPA_TDNN"
_C.MODEL.IMPUT = "fbank"
_C.MODEL.N_MELS = 80
_C.MODEL.CHANNELS = 512
_C.MODEL.NEMB = 192
_C.MODEL.RESUME = ''

# -----------------------------------------------------------------------------
# Loss settings
# -----------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.NAME = "AamSoftmax"
_C.LOSS.MARGIN = 0.2
_C.LOSS.SCALE = 30
_C.LOSS.NCLASSES = 5994
_C.LOSS.ALPHA = 10  # nlossS + alpha * nlossP 用于平衡两个loss，分类loss和对比学习loss
_C.LOSS.LAM = 0.1  # softmaxloss 和 centerloss的调节超参

# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "adam"
_C.OPTIMIZER.SCHEDULER = "CyclicLR"
_C.OPTIMIZER.LR = 0.001
_C.OPTIMIZER.BASE_LR = 1e-8
_C.OPTIMIZER.MAX_LR = 0.001
_C.OPTIMIZER.LR_DECAY = 0.97
# -----------------------------------------------------------------------------
# Optimizer settings
# -----------------------------------------------------------------------------
_C.EVALUATION = CN()
_C.EVALUATION.DCF_P_TARGET = 0.05
_C.EVALUATION.DCF_C_MISS = 1
_C.EVALUATION.DCF_C_FA = 1

# -----------------------------------------------------------------------------
# Other settings
# -----------------------------------------------------------------------------
_C.EVAL = False
_C.EVAL_MODEL = ''
_C.WANDB = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL = True
    if args.eval_model:
        config.EVAL_MODEL = args.eval_model
    if args.wandb:
        config.WANDB = True

    config.SAVE_DIR = os.path.join(config.SAVE_DIR, config.NAME + datetime.now().strftime('%Y%m%d'))
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config, args
