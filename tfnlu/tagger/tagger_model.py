import tensorflow as tf
from tfnlu.utils.encoder_model import get_encoder

from .crf import CRF, crf_loss
from .to_tags import ToTags
from .to_tokens import ToTokens


def get_lengths(x):
    return tf.reduce_sum(
        tf.clip_by_value(
            tf.strings.length(x, 'UTF8_CHAR'),
            0,
            1
        ),
        axis=-1
    )


class TaggerModel(tf.keras.Model):

    def __init__(self,
                 encoder_path,
                 word_index,
                 index_word,
                 encoder_trainable=False,
                 hidden_size=768,
                 dropout=.25,
                 n_layers=1,
                 **kwargs):
        super(TaggerModel, self).__init__(**kwargs)
        self.to_token = ToTokens(word_index)
        self.to_tags = ToTags(index_word)
        self.encoder_layer = get_encoder(encoder_path, encoder_trainable)
        self.masking = tf.keras.layers.Masking()
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.rnn_layers = []
        for _ in range(n_layers):
            rnn = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size,
                                     return_sequences=True))
            self.rnn_layers.append(rnn)
        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.project_layer = tf.keras.layers.Dense(len(word_index))
        self.crf_layer = CRF(len(word_index))

        self._set_inputs(
            tf.keras.backend.placeholder((None, None), dtype='string'))

    def compute(self, inputs, training=False):

        x = inputs

        x = self.encoder_layer(x, training=training)
        x = self.masking(x, training=training)
        x = self.dropout_layer(x, training=training)

        lengths = get_lengths(inputs)

        for rnn in self.rnn_layers:
            x = rnn(x, training=training)
        x = self.norm_layer(x, training=training)

        x = self.project_layer(x, training=training)
        return x, lengths

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        x, lengths = self.compute(inputs, training=training)
        tags_id = self.crf_layer(
            x, lengths=lengths, training=training)
        tags_id = tags_id[:, 1:-1]
        return self.to_tags(tags_id)

    def train_step(self, data):
        x, y = data
        yseq = self.to_token(y)
        transition_params = self.crf_layer.transition_params
        with tf.GradientTape() as tape:
            logits, lengths = self.compute(x, training=True)
            loss = crf_loss(inputs=logits,
                            tag_indices=yseq,
                            transition_params=transition_params,
                            lengths=lengths)
        gradients = tape.gradient(loss, self.trainable_variables)

        encoder_layers = len(self.encoder_layer.trainable_variables)
        if hasattr(self, 'optimizer_encoder'):
            self.optimizer_encoder.apply_gradients(
                zip(gradients[:encoder_layers],
                    self.trainable_variables[:encoder_layers])
            )
            self.optimizer.apply_gradients(
                zip(gradients[encoder_layers:],
                    self.trainable_variables[encoder_layers:])
            )
        else:
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

        ret = {
            m.name: m.result() for m in self.metrics
        }
        ret['loss'] = loss
        return ret
