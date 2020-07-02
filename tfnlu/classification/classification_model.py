import tensorflow as tf
import tensorflow_hub as hub

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


class ClassificationModel(tf.keras.Model):

    def __init__(self,
                 encoder_path,
                 word_index,
                 index_word,
                 encoder_trainable=False,
                 hidden_size=768,
                 dropout=.25,
                 n_layers=1,
                 **kwargs):
        super(ClassificationModel, self).__init__(**kwargs)
        self.encoder_trainable = encoder_trainable
        self.to_token = ToTokens(word_index)
        self.to_tags = ToTags(index_word)
        self.encoder_layer = hub.KerasLayer(
            encoder_path,
            trainable=encoder_trainable,
            output_key='sequence_output'
        )
        self.masking = tf.keras.layers.Masking()
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        if not self.encoder_trainable:
            self.rnn_layers = []
            for _ in range(n_layers):
                rnn = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(hidden_size,
                                         return_sequences=False))
                self.rnn_layers.append(rnn)
            self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-9)
        self.project_layer = tf.keras.layers.Dense(len(word_index))

    def compute(self, inputs, training=False):

        x = inputs

        x = self.encoder_layer(x, training=training)
        x = self.masking(x)

        x = self.dropout_layer(x, training=training)

        if not self.encoder_trainable:
            for rnn in self.rnn_layers:
                x = rnn(x, training=training)
                x = self.norm_layer(x)
        else:
            x = x[:, 0, :]

        x = self.project_layer(x, training=training)
        lengths = get_lengths(inputs)
        tags_id = tf.math.argmax(x, -1)
        return x, lengths, tags_id

    def call(self, inputs, training=False):
        _, lengths, tags_id = self.compute(inputs, training=training)
        return self.to_tags(tags_id)

    def train_step(self, data):
        x, y = data
        y = self.to_token(y)
        with tf.GradientTape() as tape:
            logits, lengths, tags_id = self.compute(x, training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, tags_id)
        ret = {
            m.name: m.result() for m in self.metrics
        }
        ret['loss'] = loss
        return ret
