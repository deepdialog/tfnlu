import tensorflow as tf
import tensorflow_hub as hub

from .crf import CRF


@tf.function(experimental_relax_shapes=True)
def get_lengths(x):
    # +2 because encoder include [CLS]/<sos> and [SEP]/<eos>
    return tf.reshape(tf.strings.length(x, 'UTF8_CHAR'),
                      (-1, )) + 2


@tf.function(experimental_relax_shapes=True)
def get_mask(x):
    return tf.sequence_mask(x)


def build_model(encoder_path,
                tag_size,
                index_word,
                encoder_trainable=True,
                hidden_size=768,
                dropout=.5,
                layers=1):
    input_layer = tf.keras.Input(shape=(1, ), dtype=tf.string)
    lengths = tf.keras.layers.Lambda(get_lengths)(input_layer)
    mask = tf.keras.layers.Lambda(get_mask)(lengths)
    x = input_layer
    _, x = hub.KerasLayer(encoder_path, trainable=encoder_trainable)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    for _ in range(layers):
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size,
                                 return_sequences=True))(x, mask=mask)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(tag_size)(x)
    tags_id, transition_params = CRF(tag_size)(x, lengths=lengths)
    model = tf.keras.Model(inputs=input_layer,
                           outputs=[tags_id, x, transition_params])
    return model
