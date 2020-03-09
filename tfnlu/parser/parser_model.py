import tensorflow as tf
from .biaffine import Biaffine


class ParserModel(tf.keras.layers.Layer):
    def __init__(self, proj0_size, proj1_size, tag0_size, tag1_size, **kwargs):
        super(ParserModel, self).__init__(**kwargs)
        self.proj0_size = proj0_size
        self.proj1_size = proj1_size
        self.tag0_size = tag0_size
        self.tag1_size = tag1_size
        self.p0 = None
        self.p1 = None
        self.b0 = None
        self.b1 = None

    def build(self, input_shape):
        self.p0 = tf.keras.layers.Dense(self.proj0_size)
        self.p1 = tf.keras.layers.Dense(self.proj1_size)
        self.b0 = Biaffine(self.proj0_size, self.proj1_size, self.tag0_size)
        self.b1 = Biaffine(self.proj0_size, self.proj1_size, self.tag1_size)
        super(ParserModel, self).build(input_shape)

    def call(self, inputs):
        """
        input:
            inputs: [batch_size, lengths]
        """
        # x0: [batch_size, lengths, proj0_size]
        x0 = self.p0(inputs)
        # x1: [batch_size, lengths, proj1_size]
        x1 = self.p1(inputs)
        # t0: [batch_size, lengths, lengths, tag0_size]
        t0 = self.b0([x0, x1])
        # t1: [batch_size, lengths, lengths, tag1_size]
        t1 = self.b1([x0, x1])
        return t0, t1
