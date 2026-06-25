import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.pos_embedding = tf.keras.layers.Embedding(seq_len, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.seq_len, delta=1)
        return x + self.pos_embedding(positions)


class TextTransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int, dropout_rate: float):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position = PositionalEmbedding(seq_len, embed_dim)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim // 4)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(embed_dim * 2, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.position(x)
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.dropout(ffn_out, training=training))
        return self.pool(x)


class SequenceEncoder(tf.keras.layers.Layer):
    def __init__(self, feature_dim: int, embed_dim: int, dropout_rate: float):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=3, padding="same", activation="relu")
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.proj = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.pool(x)
        x = self.norm(x)
        x = self.dropout(x, training=training)
        return self.proj(x)


class CnnBiLstmEncoder(tf.keras.layers.Layer):
    def __init__(self, feature_dim: int, embed_dim: int, dropout_rate: float):
        super().__init__()
        self.conv = tf.keras.layers.Conv1D(filters=embed_dim, kernel_size=5, padding="same", activation="relu")
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(embed_dim // 2, return_sequences=True)
        )
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.proj = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bilstm(x, training=training)
        x = self.pool(x)
        x = self.norm(x)
        x = self.dropout(x, training=training)
        return self.proj(x)


def build_text_encoder(vocab_size: int, seq_len: int, embed_dim: int, dropout_rate: float):
    inputs = tf.keras.Input(shape=(seq_len,), dtype=tf.int32, name="text_tokens")
    outputs = TextTransformerEncoder(vocab_size, seq_len, embed_dim, dropout_rate)(inputs)
    return tf.keras.Model(inputs, outputs, name="text_encoder")


def build_eeg_encoder(seq_len: int, feature_dim: int, embed_dim: int, dropout_rate: float):
    inputs = tf.keras.Input(shape=(seq_len, feature_dim), dtype=tf.float32, name="eeg_seq")
    outputs = CnnBiLstmEncoder(feature_dim, embed_dim, dropout_rate)(inputs)
    return tf.keras.Model(inputs, outputs, name="eeg_encoder")


def build_wearables_encoder(seq_len: int, feature_dim: int, embed_dim: int, dropout_rate: float):
    inputs = tf.keras.Input(shape=(seq_len, feature_dim), dtype=tf.float32, name="wear_seq")
    outputs = CnnBiLstmEncoder(feature_dim, embed_dim, dropout_rate)(inputs)
    return tf.keras.Model(inputs, outputs, name="wearables_encoder")


def build_audiovideo_encoder(seq_len: int, feature_dim: int, embed_dim: int, dropout_rate: float):
    inputs = tf.keras.Input(shape=(seq_len, feature_dim), dtype=tf.float32, name="av_seq")
    outputs = CnnBiLstmEncoder(feature_dim, embed_dim, dropout_rate)(inputs)
    return tf.keras.Model(inputs, outputs, name="audiovideo_encoder")
