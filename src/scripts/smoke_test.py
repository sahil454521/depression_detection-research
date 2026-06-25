import numpy as np

from src.config import ModelConfig
from src.models.model import build_model


def main():
    config = ModelConfig()
    model = build_model(config)

    batch = 2
    text = np.random.randint(0, config.text_vocab_size, size=(batch, config.text_seq_len))
    eeg = np.random.randn(batch, config.eeg_seq_len, config.eeg_feature_dim).astype("float32")
    wear = np.random.randn(batch, config.wear_seq_len, config.wear_feature_dim).astype("float32")
    av = np.random.randn(batch, config.av_seq_len, config.av_feature_dim).astype("float32")
    availability = np.ones((batch, len(config.modality_order)), dtype="float32")

    outputs = model([text, eeg, wear, av, availability], training=False)
    for name, out in outputs.items():
        print(name, out.shape)


if __name__ == "__main__":
    main()
