import tensorflow as tf


class ToTags(tf.keras.layers.Layer):
    def __init__(self, index_word, **kwargs):
        super(ToTags, self).__init__(**kwargs)
        keys = tf.constant(list(index_word.keys()), dtype=tf.int32)
        values = tf.constant(list(index_word.values()), dtype=tf.string)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            tf.constant(''))  # default value

    def call(self, inputs):
        x = inputs
        x = tf.cast(x, tf.int32)
        x = self.table.lookup(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
