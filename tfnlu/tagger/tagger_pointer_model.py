import tensorflow as tf
from tfnlu.utils.encoder_model import get_encoder

from .to_pointer_tags import ToPointerTags
from .to_pointer_tokens import ToPointerTokens


def get_lengths(x):
    return tf.reduce_sum(
        tf.clip_by_value(
            tf.strings.length(x, 'UTF8_CHAR'),
            0,
            1
        ),
        axis=-1
    )


class TaggerPointerModel(tf.keras.Model):

    def __init__(self,
                 encoder_path,
                 word_index,
                 index_word,
                 encoder_trainable=False,
                 hidden_size=768,
                 dropout=.25,
                 n_layers=1,
                 **kwargs):
        super(TaggerPointerModel, self).__init__(**kwargs)
        self.to_token = ToPointerTokens(word_index)
        self.to_tags = ToPointerTags(index_word)
        self.encoder_layer = get_encoder(encoder_path, encoder_trainable)
        self.masking = tf.keras.layers.Masking()
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        # self.rnn_layers = []
        # for _ in range(n_layers):
        #     rnn = tf.keras.layers.Bidirectional(
        #         tf.keras.layers.LSTM(hidden_size,
        #                              return_sequences=True))
        #     self.rnn_layers.append(rnn)
        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-9)

        b_keys = sorted([
            x for x in word_index.keys()
            if x.startswith('B')
        ])
        i_keys = sorted([
            x for x in word_index.keys()
            if x.startswith('I')
        ])
        assert len(b_keys) == len(i_keys)

        self.project_hidden_layers = []
        for _ in range(len(b_keys)):
            self.project_hidden_layers.append(
                tf.keras.layers.Dense(hidden_size, activation='tanh'))

        self.project_second_hidden_layers = []
        for _ in range(len(b_keys)):
            self.project_second_hidden_layers.append(
                tf.keras.layers.Dense(hidden_size, activation='tanh'))

        self.project_begin_layers = []
        for _ in range(len(b_keys)):
            self.project_begin_layers.append(tf.keras.layers.Dense(1))

        self.project_end_layers = []
        for _ in range(len(i_keys)):
            self.project_end_layers.append(tf.keras.layers.Dense(1))

        self._set_inputs(
            tf.keras.backend.placeholder((None, None), dtype='string'))

    def compute(self, inputs, training=False):

        x = inputs

        x = self.encoder_layer(x, training=training)
        x = self.masking(x, training=training)
        x = self.dropout_layer(x, training=training)

        # for rnn in self.rnn_layers:
        #     x = rnn(x, training=training)
        #     x = self.norm_layer(x, training=training)

        hiddens = [
            self.norm_layer(layer(x))
            for layer in self.project_hidden_layers
        ]
        hiddens = [
            self.norm_layer(layer(hiddens[i]))
            for i, layer in enumerate(self.project_second_hidden_layers)
        ]

        ret = []

        for i, layer in enumerate(self.project_begin_layers):
            ret.append(layer(hiddens[i]))

        for i, layer in enumerate(self.project_end_layers):
            z = tf.concat([x, hiddens[i], ret[i]], axis=-1)
            ret.append(layer(z))

        ret = tf.stack(ret)
        ret = tf.nn.sigmoid(ret)
        ret = tf.squeeze(ret, -1)
        lengths = get_lengths(inputs)
        return ret, lengths

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        x, _ = self.compute(inputs, training=training)
        x = tf.cast(x > 0.5, tf.int32)
        return self.to_tags(x)[:, 1:-1]

    def train_step(self, data):
        x, y = data
        yseq = self.to_token(y)
        with tf.GradientTape() as tape:
            logits, lengths = self.compute(x, training=True)
            loss = tf.keras.backend.binary_crossentropy(
                target=yseq, output=logits, from_logits=False
            )
            mask = tf.sequence_mask(lengths)
            mask = tf.expand_dims(mask, 0)
            mask = tf.cast(mask, tf.float32)
            loss *= mask

            loss = tf.reduce_sum(loss, -1)
            loss = tf.reduce_sum(loss, 0)
            loss = tf.reduce_sum(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        ret = {
            m.name: m.result() for m in self.metrics
        }
        ret['loss'] = loss
        return ret
