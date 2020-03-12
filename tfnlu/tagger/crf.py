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

    def call(self, inputs, lengths=None):
        """
        parameters:
            inputs [B, L, T]
            lengths [B]
        returns: [B, L]
        """
        # inputs = inputs[:, 1:-1, :]
        if lengths is None:
            shape = tf.shape(inputs)
            lengths = tf.ones((shape[0], ), dtype=tf.int32) * shape[1]
        tags_id, _ = crf_decode(
            potentials=inputs,
            transition_params=self.transition_params,
            sequence_length=lengths)
        return tags_id, tf.identity(self.transition_params)

    def get_config(self):
        return {
            'tag_size': self.tag_size,
        }


@tf.function(experimental_relax_shapes=True)
def crf_loss(inputs, tag_indices, transition_params):
    """
    parameters:
        inputs [B, L, N]
        lengths [B]
        tags [B, L, N]
    returns: loss
    """
    # +2 because encoder include [CLS]/<sos> and [SEP]/<eos>
    lengths = tf.reshape(
        tf.math.count_nonzero(tag_indices, axis=-1), (-1, ))
    # lengths = tf.add(lengths, tf.constant(2, dtype=tf.int64))
    sequence_log_likelihood, _ = crf_log_likelihood(
        inputs=inputs,
        tag_indices=tag_indices,
        sequence_lengths=lengths,
        transition_params=transition_params)
    loss = tf.reduce_mean(-sequence_log_likelihood)
    return loss
