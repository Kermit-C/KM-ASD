from torch import optim


def build_scheduler(cfg, optimizer):
    """
    Build scheduler.
    """
    if cfg.OPTIMIZER.SCHEDULER == "CyclicLR":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.OPTIMIZER.BASE_LR, max_lr=cfg.OPTIMIZER.MAX_LR, mode='triangular2',
                                                step_size_up=24000, cycle_momentum=False)
    elif cfg.OPTIMIZER.SCHEDULER == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.OPTIMIZER.LR_DECAY)
    elif cfg.OPTIMIZER.SCHEDULER == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                         threshold=0.001, threshold_mode='rel',
                                                         cooldown=0, min_lr=1e-5, eps=1e-08, verbose=True)
    print("scheduler:", cfg.OPTIMIZER.SCHEDULER)

    return scheduler