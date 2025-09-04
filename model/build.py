from .mit import mit_small


def build_model(config):
    model_name = config.MODEL.NAME
    if model_name == "mit_small":
        model = mit_small(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            attn_drop_rate=config.MODEL.ATTN_DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")
    return model
