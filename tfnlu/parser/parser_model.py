import tensorflow as tf
import tensorflow_hub as hub
from .biaffine import Biaffine
from .to_tokens import ToTokens
from .pos_to_tokens import PosToTokens
from .to_tags import ToTags
from .pos_to_tags import PosToTags


@tf.function(experimental_relax_shapes=True)
def get_lengths(x):
    # +2 because encoder include [CLS]/<sos> and [SEP]/<eos>
    return tf.reshape(tf.strings.length(x, 'UTF8_CHAR'),
                      (-1, )) + 2


@tf.function(experimental_relax_shapes=True)
def get_mask(x):
    return tf.sequence_mask(x)


class ParserModel(tf.keras.Model):
    def __init__(self,
                 encoder_path,
                 tag0_size,
                 tag1_size,
                 proj0_size,
                 proj1_size,
                 word_index, index_word,
                 pos_word_index, pos_index_word,
                 encoder_trainable=False,
                 dropout=0.33,
                 n_layers=3,
                 hidden_size=400,
                 **kwargs):
        super(ParserModel, self).__init__(**kwargs)
        self.encoder_layer = hub.KerasLayer(
            encoder_path,
            trainable=encoder_trainable
        )

        self.dropout_layer = tf.keras.layers.Dropout(dropout)

        self.rnn_layers0 = []
        for _ in range(n_layers):
            rnn = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size,
                                     return_sequences=True))
            self.rnn_layers0.append(rnn)

        self.rnn_layers1 = []
        for _ in range(n_layers):
            rnn = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(hidden_size,
                                     return_sequences=True))
            self.rnn_layers1.append(rnn)

        self.dep_mlp = tf.keras.layers.Dense(hidden_size)
        self.dep_concat = tf.keras.layers.Concatenate(axis=-1)

        self.ph0 = tf.keras.layers.Dense(proj0_size)
        self.pd0 = tf.keras.layers.Dense(proj0_size)
        self.ph1 = tf.keras.layers.Dense(proj1_size)
        self.pd1 = tf.keras.layers.Dense(proj1_size)
        self.b0 = Biaffine(proj0_size, proj0_size, tag0_size)
        self.b1 = Biaffine(proj1_size, proj1_size, tag1_size)

        self.to_tokens = ToTokens(word_index)
        self.pos_to_tokens = PosToTokens(pos_word_index)

        self.to_tags = ToTags(index_word)
        self.pos_to_tags = PosToTags(pos_index_word)

        self.pos_mlp = tf.keras.layers.Dense(hidden_size)
        self.pos_proj = tf.keras.layers.Dense(len(pos_word_index))

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)

    def compute(self, inputs, training=False):
        """
        input:
            inputs: [batch_size, lengths]
        """
        lengths = tf.keras.layers.Lambda(get_lengths)(inputs)
        mask = tf.keras.layers.Lambda(get_mask)(lengths)

        _, encoder_out = self.encoder_layer(inputs, training=training)

        x = encoder_out
        x = self.dropout_layer(x, training=training)

        for rnn in self.rnn_layers0:
            x = rnn(x, mask=mask, training=training)
            x = self.dropout_layer(x, training=training)

        pos = x
        pos = self.pos_mlp(pos)
        pos = self.leaky_relu(pos)
        pos = self.pos_proj(pos)

        x = self.dep_mlp(x)
        x = self.leaky_relu(x)
        x = self.dep_concat([encoder_out, x])

        for rnn in self.rnn_layers1:
            x = rnn(x, mask=mask, training=training)
            x = self.dropout_layer(x, training=training)

        # t0: [batch_size, lengths, lengths, tag0_size]
        t0 = self.b0([
            self.dropout_layer(
                self.pd0(x, training=training),
                training=training
            ),
            self.dropout_layer(
                self.ph0(x, training=training),
                training=training
            )
        ], training=training)
        # t1: [batch_size, lengths, lengths, tag1_size]
        t1 = self.b1([
            self.dropout_layer(
                self.pd1(x, training=training),
                training=training
            ),
            self.dropout_layer(
                self.ph1(x, training=training),
                training=training
            )
        ], training=training)
        t1 = tf.nn.sigmoid(t1)
        return t0, t1, pos

    def call(self, inputs, training=False):
        """
        input:
            inputs: [batch_size, lengths]
        """
        arc, rel, pos = self.compute(inputs, training=training)

        pos = tf.argmax(pos, -1)

        # 在第二维上最大
        max_2 = tf.math.reduce_max(rel, axis=1)
        z = tf.cast(
            rel >= tf.expand_dims(max_2, 1),
            arc.dtype)
        arc *= tf.expand_dims(z, -1)
        arc = tf.argmax(arc, -1)

        arc = self.to_tags(arc)
        pos = self.pos_to_tags(pos)

        return arc, pos

    def train_step(self, data):
        x, (y0, y1) = data
        lengths = tf.strings.length(
            x, unit='UTF8_CHAR'
        ) + 2
        y_arc = self.to_tokens(y0)
        y_pos = self.pos_to_tokens(y1)
        y_rel = tf.cast(
            tf.math.greater(y_arc, 0),
            tf.float32
        )
        with tf.GradientTape() as tape:
            p_arc, p_rel, p_pos = self.compute(x, training=True)
            l0 = parser_loss(y_arc, p_arc)
            l1 = parser_loss_bin(y_rel, p_rel, lengths) * 5.0
            l2 = parser_loss_pos(y_pos, p_pos, lengths) * 0.01
            loss = l0 + l1 + l2
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        ret = {
            m.name: m.result() for m in self.metrics
        }
        ret['loss'] = loss
        ret['l0'] = l0
        ret['l1'] = l1
        ret['l2'] = l2
        return ret


def parser_loss_bin(y_true, y_pred, lengths):
    mask = tf.sequence_mask(lengths)
    mask = tf.cast(mask, tf.float32)
    mask = tf.transpose(mask, (0, 2, 1)) @ mask
    loss = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(mask), loss.dtype)
    return loss


def parser_loss_pos(y_true, y_pred, lengths):
    mask = tf.sequence_mask(lengths)
    mask = tf.cast(mask, tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def parser_loss(y_true, y_pred):
    mask = tf.cast(
        tf.math.greater(y_true, 0),
        tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    loss *= tf.cast(mask, loss.dtype)
    loss = tf.reduce_sum(loss) / tf.cast(tf.reduce_sum(mask), loss.dtype)
    return loss
