from torch import optim


def build_optimizer(cfg, model):
    """
    Build optimizer.
    """

    if cfg.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR, weight_decay=2e-5)
    elif cfg.OPTIMIZER.NAME == 'adamw':
        optimizer = optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999),
                                lr=cfg.OPTIMIZER.LR, weight_decay=0.05)
    print("optimizer:", cfg.OPTIMIZER.NAME)
    return optimizer