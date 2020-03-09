import tensorflow as tf
import tensorflow_hub as hub
from .parser_model import ParserModel


def build_model(encoder_path):
    input_layer = tf.keras.layers.Input((1,), dtype=tf.string)
    x = input_layer
    x = hub.KerasLayer(encoder_path)(x)
    x = ParserModel(50, 50, 2, 15)(x)
    model = tf.keras.Model(
        inputs=input_layer, outputs=x)
    return model
