import tensorflow as tf


class ToTokens(tf.keras.layers.Layer):
    def __init__(self, word_index, default_value=0, **kwargs):
        super(ToTokens, self).__init__(**kwargs)
        keys = tf.constant(list(word_index.keys()), dtype=tf.string)
        values = tf.constant(list(word_index.values()), dtype=tf.int32)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            tf.constant(default_value))  # default value
        self.default_value = default_value

    def call(self, inputs):
        x = inputs
        x = self.table.lookup(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], None
