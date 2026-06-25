import tensorflow as tf


class AdaptiveMultimodalFusion(tf.keras.layers.Layer):
    def __init__(self, num_modalities: int, embed_dim: int, fusion_dim: int, heads: int, key_dim: int, dropout_rate: float):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate_dense = tf.keras.layers.Dense(num_modalities)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fuse_proj = tf.keras.layers.Dense(fusion_dim, activation="relu")
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, modality_embeddings, availability_mask, training=False):
        # modality_embeddings: list of [batch, embed_dim]
        # availability_mask: [batch, num_modalities] with 1 for present, 0 for missing
        stacked = tf.stack(modality_embeddings, axis=1)
        mask = tf.cast(availability_mask, tf.float32)

        # Gating over modalities
        gate_logits = self.gate_dense(stacked)
        gate_logits = tf.reduce_mean(gate_logits, axis=-1)
        masked_logits = gate_logits + (1.0 - mask) * -1.0e9
        gate_weights = tf.nn.softmax(masked_logits, axis=-1)

        gated = tf.einsum("bm,bmd->bd", gate_weights, stacked)

        # Cross-attention across modality tokens
        attn_out = self.attn(stacked, stacked, training=training)
        attn_out = self.dropout(attn_out, training=training)
        attn_out = tf.reduce_mean(attn_out, axis=1)

        fused = tf.concat([gated, attn_out], axis=-1)
        fused = self.norm(fused)
        return self.fuse_proj(fused)
