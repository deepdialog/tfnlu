
import tensorflow as tf
import tensorflow_addons as tfa


def get_opt(optimizer, optimizer_arguments):
    optimizer = optimizer.lower()
    if optimizer == 'adamw':
        model_optimizer = tf.keras.optimizers.AdamW(
            **optimizer_arguments
        )
    elif optimizer == 'rectifiedadam':
        model_optimizer = tfa.optimizers.RectifiedAdam(
            **optimizer_arguments
        )
    else:
        model_optimizer = tf.keras.optimizers.Adam(
            **optimizer_arguments
        )
    return model_optimizer
