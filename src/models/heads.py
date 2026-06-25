import tensorflow as tf


class BinaryHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.out = tf.keras.layers.Dense(1, activation="sigmoid", name="binary_output")

    def call(self, x):
        return self.out(x)


class SeverityHead(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.out = tf.keras.layers.Dense(1, activation="linear", name="severity_output")

    def call(self, x):
        return self.out(x)


class MultiLabelHead(tf.keras.layers.Layer):
    def __init__(self, num_labels: int):
        super().__init__()
        self.out = tf.keras.layers.Dense(num_labels, activation="sigmoid", name="symptom_output")

    def call(self, x):
        return self.out(x)
