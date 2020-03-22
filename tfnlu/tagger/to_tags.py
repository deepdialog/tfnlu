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
        x, mask = inputs
        x = self.table.lookup(x)
        x = tf.ragged.boolean_mask(x, mask)
        x = tf.strings.reduce_join(
            x, axis=-1, separator=' '
        )
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
