
import math
import tensorflow as tf


def gelu(x):
    """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


class Attention(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, size_per_head, initializer_range,
                 attention_probs_dropout_prob, **kwargs):
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.query_layer = None
        self.key_layer = None
        self.value_layer = None
        # used for attention scores before softmax
        self.negative_infinity = -10000.0
        super(Attention, self).__init__(**kwargs)

    @staticmethod
    def create_attention_mask(from_shape, input_mask):
        """
        Creates 3D attention.
        :param from_shape:  [batch_size, from_seq_len, ...]
        :param input_mask:  [batch_size, seq_len]
        :return: [batch_size, from_seq_len, seq_len]
        """

        mask = tf.cast(tf.expand_dims(input_mask, axis=1),
                       tf.keras.backend.floatx())  # [B, 1, T]
        ones = tf.expand_dims(tf.ones(shape=from_shape[:2],
                                      dtype=tf.keras.backend.floatx()),
                              axis=-1)  # [B, F, 1]
        mask = ones * mask  # broadcast along two dimensions

        return mask  # [B, F, T]

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        dense_units = self.num_attention_heads * self.size_per_head  # N*H
        #
        # B, F, T, N, H
        # batch, from_seq_len, to_seq_len, num_attention_heads, size_per_head
        #
        self.query_layer = tf.keras.layers.Dense(
            units=dense_units,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range),
            name="query")
        self.key_layer = tf.keras.layers.Dense(
            units=dense_units,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range),
            name="key")
        self.value_layer = tf.keras.layers.Dense(
            units=dense_units,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range),
            name="value")
        self.dropout_layer = tf.keras.layers.Dropout(
            self.attention_probs_dropout_prob)

        super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        from_shape = input_shape

        # from_shape         # [B, F, W]
        # [batch_size, from_seq_length, from_width]
        # input_mask_shape   # [B, F]

        output_shape = [
            from_shape[0], from_shape[1],
            self.num_attention_heads * self.size_per_head
        ]

        return output_shape  # [B, F, N*H]

    # noinspection PyUnusedLocal
    def call(self, inputs, mask=None, training=None, **kwargs):
        from_tensor = inputs
        to_tensor = inputs
        if mask is None:
            mask = tf.ones(tf.shape(from_tensor)[:2], dtype=tf.int32)
        attention_mask = Attention.create_attention_mask(
            tf.shape(input=from_tensor), mask)

        #  from_tensor shape - [batch_size, from_seq_length, from_width]
        input_shape = tf.shape(input=from_tensor)
        batch_size, from_seq_len = input_shape[0], input_shape[1]
        to_seq_len = from_seq_len

        # [B, F, N*H] -> [B, N, F, H]
        def transpose_for_scores(input_tensor, seq_len):
            output_shape = [
                batch_size, seq_len, self.num_attention_heads,
                self.size_per_head
            ]
            output_tensor = tf.reshape(input_tensor, output_shape)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1,
                                                       3])  # [B,N,F,H]

        query = self.query_layer(
            from_tensor)  # [B,F, N*H] [batch_size, from_seq_len, N*H]
        key = self.key_layer(to_tensor)  # [B,T, N*H]
        value = self.value_layer(to_tensor)  # [B,T, N*H]

        query = transpose_for_scores(query, from_seq_len)  # [B, N, F, H]
        key = transpose_for_scores(key, to_seq_len)  # [B, N, T, H]

        attention_scores = tf.matmul(query, key,
                                     transpose_b=True)  # [B, N, F, T]
        size_per_head = tf.constant(self.size_per_head,
                                    dtype=tf.keras.backend.floatx())
        attention_scores = attention_scores / tf.sqrt(size_per_head)

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask,
                                            axis=1)  # [B, 1, F, T]
            # {1, 0} -> {0.0, -inf}
            adder = (1.0 - tf.cast(attention_mask, tf.keras.backend.floatx())
                     ) * self.negative_infinity
            attention_scores = tf.add(
                attention_scores,
                adder)  # adding to softmax -> its like removing them entirely

        # scores to probabilities
        attention_probs = tf.nn.softmax(attention_scores)  # [B, N, F, T]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout_layer(attention_probs,
                                             training=training)  # [B, N, F, T]

        # [B,T,N,H]
        value = tf.reshape(value, [
            batch_size, to_seq_len, self.num_attention_heads,
            self.size_per_head
        ])
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])  # [B, N, T, H]

        context_layer = tf.matmul(attention_probs, value)  # [B, N, F, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1,
                                                            3])  # [B, F, N, H]

        output_shape = [
            batch_size, from_seq_len,
            self.num_attention_heads * self.size_per_head
        ]
        context_layer = tf.reshape(context_layer, output_shape)
        return context_layer  # [B, F, N*H]

    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        return mask  # [B, F]


class Projection(tf.keras.layers.Layer):
    def __init__(self, hidden_size, hidden_dropout_prob, initializer_range,
                 **kwargs):
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.dense = None
        self.dropout = None
        self.layer_norm = None
        super(Projection, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and 2 == len(input_shape)
        out_shape, residual_shape = input_shape
        self.input_spec = [
            tf.keras.layers.InputSpec(shape=out_shape),
            tf.keras.layers.InputSpec(shape=residual_shape)
        ]

        self.dense = tf.keras.layers.Dense(
            units=self.hidden_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range),
            name="dense")
        self.dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)

        super(Projection, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        output, residual = inputs
        output = self.dense(output)
        output = self.dropout(output, training=training)
        output = tf.add(output, residual)
        return output


class TransformerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob,
                 initializer_range, attention_probs_dropout_prob, **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range

        self.size_per_head = hidden_size // num_attention_heads
        self.attention_layer = None
        self.attention_projector = None

        super(TransformerSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        self.attention_layer = Attention(
            num_attention_heads=self.num_attention_heads,
            size_per_head=self.size_per_head,
            initializer_range=self.initializer_range,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            name="self",
        )
        self.attention_projector = Projection(
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            name="output")

        super(TransformerSelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        x = inputs
        attention_head = self.attention_layer(x, mask=mask, training=training)
        x = self.attention_projector([attention_head, x],
                                     mask=mask,
                                     training=training)

        return x


class Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 intermediate_size=768 * 4,
                 hidden_act='gelu',
                 initializer_range=0.02,
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                 return_sequences=True, **kwargs):
        assert hidden_size % num_attention_heads == 0
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.size_per_head = hidden_size // num_attention_heads
        self.self_attention_layer = None
        self.intermediate_layer = None
        self.output_projector = None

        super(Transformer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(
            shape=input_shape)  # [B, seq_len, hidden_size]

        self.self_attention_layer = TransformerSelfAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            name="attention")

        self.intermediate_layer = tf.keras.layers.Dense(
            name="intermediate/dense",
            units=self.intermediate_size,
            activation=gelu if self.hidden_act == 'gelu' else self.hidden_act,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=self.initializer_range))

        self.output_projector = Projection(
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            name="output")

        # epsilon is important be same with tf.contrib.layers.layer_norm
        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/layers/python/layers/layers.py
        # L2174
        self.layer_norm_attention = tf.keras.layers.LayerNormalization(
            epsilon=1e-12,
            name='attention/output/LayerNorm')
        self.layer_norm_output = tf.keras.layers.LayerNormalization(
            epsilon=1e-12,
            name='output/LayerNorm')

        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        layer_input = inputs

        attention_output = self.self_attention_layer(layer_input,
                                                     mask=mask,
                                                     training=training)
        attention_output = self.layer_norm_attention(attention_output)
        # intermediate
        intermediate_output = self.intermediate_layer(attention_output)
        # output
        layer_output = self.output_projector(
            [intermediate_output, attention_output],
            mask=mask,
            training=training)
        layer_output = self.layer_norm_output(layer_output)

        return layer_output


if __name__ == "__main__":
    t = Transformer()
    a = tf.random.uniform((2, 3, 768))
    b = t(a)
    print(a.shape)
    print(b.shape)
    print(a[0][0][:3])
    print(b[0][0][:3])
