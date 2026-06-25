import tensorflow as tf


def gradient_reversal(x, scale=1.0):
    @tf.custom_gradient
    def _grl(t):
        def grad(dy):
            return -scale * dy
        return t, grad

    return _grl(x)


class DomainDiscriminator(tf.keras.layers.Layer):
    def __init__(self, num_domains: int, hidden_dim: int = 64):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(num_domains, activation="softmax")

    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)


def apply_domain_adaptation(fused_repr, num_domains: int, grl_scale: float):
    reversed_repr = gradient_reversal(fused_repr, scale=grl_scale)
    discriminator = DomainDiscriminator(num_domains)
    return discriminator(reversed_repr)
