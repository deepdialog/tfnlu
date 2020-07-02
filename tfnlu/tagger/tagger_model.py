import tensorflow as tf
import tensorflow_hub as hub

from .crf import CRF, crf_loss
from .to_tags import ToTags
from .to_tokens import ToTokens


@tf.function
def get_lengths(x):
    return tf.reduce_sum(
        tf.clip_by_value(
            tf.strings.length(x, 'UTF8_CHAR'),
            0,
            1
        ),
        axis=-1
    )


@tf.function
def get_mask(x):
    x, maxlen = x
    return tf.sequence_mask(x, maxlen=maxlen)


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
        self.encoder_layer = hub.KerasLayer(
            encoder_path,
            trainable=encoder_trainable,
            output_key='sequence_output'
        )
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

    def compute(self, inputs, training=False):

        x = inputs

        x = self.encoder_layer(x, training=training)
        x = self.masking(x, training=training)
        x = self.dropout_layer(x, training=training)

        lengths = get_lengths(inputs)

        for rnn in self.rnn_layers:
            x = rnn(x, training=training)
            x = self.norm_layer(x, training=training)
            # x = self.dropout_layer(x, training=training)

        x = self.project_layer(x, training=training)
        tags_id = self.crf_layer(
            x, lengths=lengths, training=training)
        tags_id = tags_id[:, 1:-1]
        return x, lengths, tags_id

    def call(self, inputs, training=False):
        _, lengths, tags_id = self.compute(inputs, training=training)
        mask = tf.keras.layers.Lambda(get_mask)([
            lengths - 2,
            tf.shape(tags_id)[1]
        ])
        return self.to_tags([tags_id, mask])

    def train_step(self, data):
        x, y = data
        yseq = self.to_token(y)
        with tf.GradientTape() as tape:
            logits, lengths, tags_id = self.compute(x, training=True)
            transition_params = self.crf_layer.transition_params
            loss = crf_loss(inputs=logits,
                            tag_indices=yseq,
                            transition_params=transition_params,
                            lengths=lengths)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, tags_id)
        ret = {
            m.name: m.result() for m in self.metrics
        }
        ret['loss'] = loss
        return ret
