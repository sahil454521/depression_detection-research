import tensorflow as tf

from src.config import ModelConfig
from src.models.encoders import (
    build_text_encoder,
    build_eeg_encoder,
    build_wearables_encoder,
    build_audiovideo_encoder,
)
from src.models.fusion import AdaptiveMultimodalFusion
from src.models.heads import BinaryHead, SeverityHead, MultiLabelHead


def build_model(config: ModelConfig) -> tf.keras.Model:
    text_input = tf.keras.Input(shape=(config.text_seq_len,), dtype=tf.int32, name="text_tokens")
    eeg_input = tf.keras.Input(
        shape=(config.eeg_seq_len, config.eeg_feature_dim), dtype=tf.float32, name="eeg_seq"
    )
    wear_input = tf.keras.Input(
        shape=(config.wear_seq_len, config.wear_feature_dim), dtype=tf.float32, name="wear_seq"
    )
    av_input = tf.keras.Input(
        shape=(config.av_seq_len, config.av_feature_dim), dtype=tf.float32, name="av_seq"
    )
    availability = tf.keras.Input(shape=(len(config.modality_order),), dtype=tf.float32, name="availability")

    text_encoder = build_text_encoder(
        config.text_vocab_size, config.text_seq_len, config.embed_dim, config.dropout_rate
    )
    eeg_encoder = build_eeg_encoder(
        config.eeg_seq_len, config.eeg_feature_dim, config.embed_dim, config.dropout_rate
    )
    wear_encoder = build_wearables_encoder(
        config.wear_seq_len, config.wear_feature_dim, config.embed_dim, config.dropout_rate
    )
    av_encoder = build_audiovideo_encoder(
        config.av_seq_len, config.av_feature_dim, config.embed_dim, config.dropout_rate
    )

    embeddings = [
        text_encoder(text_input),
        eeg_encoder(eeg_input),
        wear_encoder(wear_input),
        av_encoder(av_input),
    ]

    fusion = AdaptiveMultimodalFusion(
        num_modalities=len(config.modality_order),
        embed_dim=config.embed_dim,
        fusion_dim=config.fusion_dim,
        heads=config.mha_heads,
        key_dim=config.mha_key_dim,
        dropout_rate=config.dropout_rate,
    )

    fused = fusion(embeddings, availability)

    binary = BinaryHead()(fused)
    severity = SeverityHead()(fused)
    multi = MultiLabelHead(config.num_symptoms)(fused)

    return tf.keras.Model(
        inputs=[text_input, eeg_input, wear_input, av_input, availability],
        outputs={"binary": binary, "severity": severity, "symptoms": multi},
        name="adaptive_multimodal_model",
    )
