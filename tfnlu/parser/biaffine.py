import tensorflow as tf


class Biaffine(tf.keras.layers.Layer):
    def __init__(self,
                 in_dim0, in_dim1,
                 out_dim=1,
                 bias0=True, bias1=True, **kwargs):
        super(Biaffine, self).__init__(**kwargs)
        self.in_dim0 = in_dim0
        self.in_dim1 = in_dim1
        self.out_dim = out_dim
        self.bias0 = bias0
        self.bias1 = bias1

    def build(self, input_shape):
        self.w = self.add_weight(
            initializer=tf.keras.initializers.glorot_uniform(),
            shape=(
                self.out_dim,
                self.in_dim0 + (1 if self.bias0 else 0),
                self.in_dim1 + (1 if self.bias1 else 0)
            ),
            dtype=tf.dtypes.float32,
            name='biaffine_matrix')
        super(Biaffine, self).build(input_shape)

    def call(self, inputs):
        """
        B: batch size
        L: sentence lengths
        D0: in_dim0 in constructor
        D1: in_dim1 in constructor
        OD: out_dim in constructor
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
        assert x0.shape[2] == self.in_dim0
        assert x1.shape[2] == self.in_dim1
        if self.bias0:
            x0 = tf.concat([
                x0,
                tf.ones_like(x0[:, :, :1])
            ], axis=-1)
        if self.bias1:
            x1 = tf.concat([
                x1,
                tf.ones_like(x0[:, :, :1])
            ], axis=-1)
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
