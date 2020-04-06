import tensorflow as tf
import tensorflow_hub as hub

from .crf import CRF, crf_loss
from .to_tags import ToTags
from .to_tokens import ToTokens


@tf.function(experimental_relax_shapes=True)
def get_lengths(x):
    # +2 because encoder include [CLS]/<sos> and [SEP]/<eos>
    return tf.reshape(tf.strings.length(x, 'UTF8_CHAR'),
                      (-1, )) + 2


@tf.function(experimental_relax_shapes=True)
def get_mask(x):
    return tf.sequence_mask(x)


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
            trainable=encoder_trainable
        )
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.rnn_layers = []
        for _ in range(n_layers):
            rnn = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size,
                                     return_sequences=True))
            self.rnn_layers.append(rnn)
        self.project_layer = tf.keras.layers.Dense(len(word_index))
        self.crf_layer = CRF(len(word_index))

    def compute(self, inputs, training=False):
        lengths = tf.keras.layers.Lambda(get_lengths)(inputs)
        mask = tf.keras.layers.Lambda(get_mask)(lengths)

        x = inputs

        _, x = self.encoder_layer(x, training=training)

        x = self.dropout_layer(x, training=training)

        for rnn in self.rnn_layers:
            x = rnn(x, mask=mask, training=training)
            x = self.dropout_layer(x, training=training)

        x = self.project_layer(x, training=training)
        tags_id = self.crf_layer(
            x, lengths=lengths, training=training)
        tags_id = tags_id[:, 1:-1]
        return x, lengths, tags_id

    def call(self, inputs, training=False):
        _, lengths, tags_id = self.compute(inputs, training=training)
        mask = tf.keras.layers.Lambda(get_mask)(lengths - 2)
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
