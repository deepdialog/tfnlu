import os
import tensorflow as tf
import tensorflow_hub as hub
from .logger import logger


class UnicodeEncoder(tf.keras.Model):
    def __init__(self, embedding_size=128, vocab_size_max=65536, **kwargs):
        super(UnicodeEncoder, self).__init__(**kwargs)
        self.preprocess_layer = tf.keras.layers.Lambda(
            lambda x: tf.strings.substr(tf.strings.regex_replace(
                tf.strings.regex_replace(x, r'\[CLS\]', '\a'), r'\[SEP\]', '\b'
            ),
                                        0,
                                        1,
                                        unit='UTF8_CHAR'))
        self.encode_layer = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(
            tf.squeeze(tf.strings.unicode_decode(x, 'UTF-8').to_tensor(), -1),
            0, vocab_size_max - 1))
        self.embedding_layer = tf.keras.layers.Embedding(
            vocab_size_max, embedding_size)

        self.masking_layer = tf.keras.layers.Lambda(
            lambda x: x[0] * tf.expand_dims(
                tf.cast(tf.clip_by_value(tf.strings.length(x[1]), 0, 1), tf.
                        float32), -1))

        self._set_inputs(tf.keras.backend.placeholder((None, ),
                                                      dtype='string'))

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.string)])
    def call(self, inputs):
        x = inputs
        x = self.preprocess_layer(x)
        x = self.encode_layer(x)
        x = self.embedding_layer(x)
        x = self.masking_layer([x, inputs])
        return x


def get_encoder(encoder_path, encoder_trainable):
    if encoder_path is not None:
        if not os.path.exists(encoder_path):
            raise RuntimeError('No encoder_path found')
        return hub.KerasLayer(encoder_path,
                              trainable=encoder_trainable,
                              output_key='sequence_output')
    else:
        logger.warn(
            'No encoder_path define, use default encoder, '
            'if you need better performance, '
            'you could try download a BERT encoder from: '
            'https://github.com/qhduan/bert-model'
        )
        return UnicodeEncoder()
