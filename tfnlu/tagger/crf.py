import tensorflow as tf
from tensorflow_addons.text.crf import crf_decode, crf_log_likelihood


class CRF(tf.keras.layers.Layer):
    def __init__(self, tag_size, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.tag_size = tag_size

    def build(self, input_shape):

        self.transition_params = self.add_weight(
            initializer=tf.keras.initializers.glorot_uniform(),
            shape=(self.tag_size, self.tag_size),
            dtype=tf.keras.backend.floatx(),
            name='crf/transition_params')
        super(CRF, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        batch_size, lengths, hidden_size = input_shape[0]
        return tf.TensorShape([batch_size, lengths])

    def call(self, inputs):
        """
        parameters:
            inputs [B, L, T]
            lengths [B]
        returns: [B, L]
        """
        inputs = inputs[:, 1:-1, :]
        shape = tf.shape(inputs)
        lengths = tf.ones((shape[0], ), dtype=tf.int32) * shape[1]
        tags_id, _ = crf_decode(inputs, self.transition_params, lengths)
        return inputs, self.transition_params, tags_id

    def get_config(self):
        return {
            'tag_size': self.tag_size,
        }


@tf.function
def crf_loss(tags, logits, transition_params):
    """
    parameters:
        inputs [B, L, N]
        lengths [B]
        tags [B, L, N]
    returns: loss
    """
    lengths = tf.reshape(tf.math.count_nonzero(tags, axis=-1), (-1, ))
    sequence_log_likelihood, _ = crf_log_likelihood(
        inputs=logits,
        tag_indices=tags,
        sequence_lengths=lengths,
        transition_params=transition_params)
    loss = tf.reduce_mean(-sequence_log_likelihood)
    return loss
