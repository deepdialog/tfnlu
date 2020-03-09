import tensorflow as tf


class Biaffine(tf.keras.layers.Layer):
    def __init__(self, dim0, dim1, odim=1, **kwargs):
        super(Biaffine, self).__init__(**kwargs)
        self.dim0 = dim0
        self.dim1 = dim1
        self.odim = odim

    def build(self, input_shape):
        self.w = self.add_weight(
            initializer=tf.keras.initializers.glorot_uniform(),
            shape=(self.odim, self.dim0, self.dim1),
            dtype=tf.dtypes.float32,
            name='biaffine_matrix')
        super(Biaffine, self).build(input_shape)

    def call(self, inputs):
        """
        B: batch size
        L: sentence lengths
        D0: dim0 in constructor
        D1: dim1 in constructor
        OD: odim in constructor
        input:
        x0: [B, L, D0]
        x1: [B, L, D1]
        output:
        [B, L, L, OD]
            or
        [B, L, L]
        """
        x0, x1 = inputs
        assert len(x0.shape) == 3
        assert len(x1.shape) == 3
        assert x0.shape[0] == x1.shape[0]
        assert x0.shape[1] == x1.shape[1]
        assert x0.shape[2] == self.dim0
        assert x1.shape[2] == self.dim1
        # x0: [B, 1, L, D0]
        x0 = tf.expand_dims(x0, 1)
        # x1: [B, 1, L, D1]
        x1 = tf.expand_dims(x1, 1)
        # x1: [B, 1, D1, L]
        x1 = tf.transpose(x1, (0, 1, 3, 2))
        # r: [B, 1, L, D0] * [OD, D0, D1] * [B, 1, D1, L]
        # r: [B, OD, L, D1] * [B, 1, D1, L]
        # r: [B OD, L, L]
        r = x0 @ self.w @ x1
        if r.shape[1] == 1:
            # [B, L, L]
            r = tf.squeeze(r, 1)
        else:
            # [B, L, L, OD]
            r = tf.transpose(r, (0, 2, 3, 1))
        return r
