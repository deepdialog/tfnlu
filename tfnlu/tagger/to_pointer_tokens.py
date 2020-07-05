import tensorflow as tf


class ToPointerTokens(tf.keras.layers.Layer):
    def __init__(self, word_index, default_value=0, **kwargs):
        super(ToPointerTokens, self).__init__(**kwargs)
        b_keys = sorted([
            x for x in word_index.keys()
            if x.startswith('B')
        ])
        i_keys = sorted([
            x for x in word_index.keys()
            if x.startswith('I')
        ])
        assert len(b_keys) == len(i_keys)
        self.b_keys = b_keys
        self.i_keys = i_keys

    def call(self, inputs):
        x = inputs
        ret = []
        for b in self.b_keys:
            z = tf.cast(x == b, tf.float32)
            ret.append(z)
        for i in self.i_keys:
            z = tf.cast(x == i, tf.float32)
            z = z * tf.abs(
                tf.concat(
                    [
                        z[:, 1:],
                        tf.zeros((tf.shape(z)[0], 1), tf.float32)
                    ], axis=1) - 1
            )
            ret.append(z)
        ret = tf.stack(ret)
        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[0], None
