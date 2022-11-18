from re import I


def get_model(cfg):
    model_name = cfg["experiment"]["id"]
    if model_name == "deeplabLarge":
        from .deeplab import DeeplabLarge

        model = DeeplabLarge(cfg)
    elif model_name == "fasterrcnn":
        from models.fasterrcnn import FasterRCNN

        model = FasterRCNN(cfg)
    else:
        raise AttributeError("Model {} not implemented".format(model_name))

    return model
