import tensorflow as tf


class PosToTokens(tf.keras.layers.Layer):
    def __init__(self, word_index, default_value=0, **kwargs):
        super(PosToTokens, self).__init__(**kwargs)
        keys = tf.constant(list(word_index.keys()), dtype=tf.string)
        values = tf.constant(list(word_index.values()), dtype=tf.int32)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            tf.constant(default_value))  # default value
        self.default_value = default_value

    def call(self, inputs):
        x = inputs
        # Add start token and remove, for better NER performance
        # x = tf.strings.regex_replace(x, '^', '[CLS] ')
        # x = tf.strings.regex_replace(x, '$', ' [SEP]')
        # Trick, reshape for TensorFlow problem
        # x = tf.reshape(x, (-1,))
        # x = tf.strings.split(x, sep=' ')
        # x = x.to_tensor('')
        x = self.table.lookup(x)
        # Trick, force TensorFlow thinks we will
        # - return [None, None], not [None,]
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], None
