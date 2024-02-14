from . import net


def built_model(cfg):
    if cfg.MODEL.NAME == "ECAPA_TDNN":
        model = getattr(net, cfg.MODEL.NAME)(in_channels=cfg.MODEL.N_MELS, channels=cfg.MODEL.CHANNELS, embd_dim=cfg.MODEL.NEMB)
    if cfg.MODEL.NAME == "ECAPA_TDNN1":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.N_MELS, cfg.MODEL.IMPUT)
    if cfg.MODEL.NAME == "ECAPA_TDNN1_xs":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.N_MELS, cfg.MODEL.IMPUT)
    if cfg.MODEL.NAME == "ECAPA_TDNN1_BN":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.N_MELS, cfg.MODEL.IMPUT)
    if cfg.MODEL.NAME == "ECAPA_TDNN1_LN":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.N_MELS, cfg.MODEL.IMPUT)
    if cfg.MODEL.NAME == "ECAPA_TDNN1_CN":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.N_MELS, cfg.MODEL.IMPUT)
    if cfg.MODEL.NAME == "ECAPA_TDNN_RNN":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.N_MELS, cfg.MODEL.IMPUT)
    if cfg.MODEL.NAME == "ECAPA_TDNN_flow":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.N_MELS, cfg.MODEL.IMPUT)
    if cfg.MODEL.NAME == "ECAPA_TDNN2":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "ECAPA_TDNN_wav":
        model = getattr(net, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "ResNetTDNN":
        model = getattr(net, cfg.MODEL.NAME)(channels=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "ECAPA_TDNN_pool":
        model = getattr(net, cfg.MODEL.NAME)(channels=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "SwinTransformer":
        model = getattr(net, cfg.MODEL.NAME)()
    if cfg.MODEL.NAME == "ResNetSE34V2":
        model = getattr(net, cfg.MODEL.NAME)(nOut=cfg.MODEL.NEMB)
    if cfg.MODEL.NAME == "SwinTransformerV1":
        model = getattr(net, cfg.MODEL.NAME)()
    if cfg.MODEL.NAME == "SwinTransformerV2":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "SwinTransformerV2_nds":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "SwinTransformerV2BN":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "SwinTransformerV2_7":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "SwinTransformerV2_25":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "WinTransformer":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "WinTransformer_nds":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    if cfg.MODEL.NAME == "SwinTransformerV11":
        model = getattr(net, cfg.MODEL.NAME)()
    if cfg.MODEL.NAME == "ViT":
        model = getattr(net, cfg.MODEL.NAME)()
    if cfg.MODEL.NAME == "CRTDNN":
        model = getattr(net, cfg.MODEL.NAME)()
    if cfg.MODEL.NAME == "HrResTDNN":
        model = getattr(net, cfg.MODEL.NAME)()
    if cfg.MODEL.NAME == "SwinTransformerV2_attnmap":
        model = getattr(net, cfg.MODEL.NAME)(in_chans=cfg.MODEL.N_MELS, embed_dim=cfg.MODEL.CHANNELS)
    print("model:", cfg.MODEL.NAME)
    return model
