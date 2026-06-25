from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    text_vocab_size: int = 30000
    text_seq_len: int = 128
    eeg_seq_len: int = 256
    eeg_feature_dim: int = 64
    wear_seq_len: int = 128
    wear_feature_dim: int = 32
    av_seq_len: int = 200
    av_feature_dim: int = 40

    embed_dim: int = 128
    fusion_dim: int = 128
    mha_heads: int = 4
    mha_key_dim: int = 32
    dropout_rate: float = 0.1

    num_symptoms: int = 9

    modality_order: tuple = ("text", "eeg", "wearables", "audiovideo")


@dataclass(frozen=True)
class CrossDatasetConfig:
    domain_adaptation: bool = True
    transfer_learning: bool = True
    lodo_validation: bool = True
    multi_center_eval: bool = True
    cross_cultural_benchmark: bool = True

    adaptation_lambda: float = 0.1
    grl_scale: float = 1.0

    source_datasets: tuple = ("DAIC-WOZ", "MODMA", "EATD", "PHQ-9")
    target_datasets: tuple = ()
    domain_keys: tuple = ("dataset", "center", "culture")
