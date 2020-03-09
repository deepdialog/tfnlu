import tensorflow as tf
import tensorflow_hub as hub

from .crf import CRF


def build_model(encoder_path, tag_size, index_word, hidden_size=768):
    input_layer = tf.keras.Input(shape=(1, ), dtype=tf.string)
    x = input_layer
    x = hub.KerasLayer(encoder_path)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size, return_sequences=True))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tag_size))(x)
    logits, transition_params, x = CRF(tag_size)(x)
    train_model = tf.keras.Model(inputs=input_layer,
                                 outputs=[x, logits, transition_params])
    predict_model = tf.keras.Model(inputs=input_layer, outputs=x)
    return train_model, predict_model
