def set_encoder_trainable(model, trainable: bool, encoder_prefixes=None):
    prefixes = encoder_prefixes or ("text_encoder", "eeg_encoder", "wearables_encoder", "audiovideo_encoder")
    for layer in model.layers:
        if any(layer.name.startswith(prefix) for prefix in prefixes):
            layer.trainable = trainable


def freeze_backbone(model, encoder_prefixes=None):
    set_encoder_trainable(model, False, encoder_prefixes=encoder_prefixes)


def unfreeze_backbone(model, encoder_prefixes=None):
    set_encoder_trainable(model, True, encoder_prefixes=encoder_prefixes)
