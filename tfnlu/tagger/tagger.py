import math
import logging

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from sklearn.utils import shuffle as skshuffle

from tfnlu.utils.serialize_dir import serialize_dir, deserialize_dir
from tfnlu.utils.temp_dir import TempDir
from .get_tags import get_tags
from .build_model import build_model
from .to_tokens import ToTokens
from .crf import crf_loss
from .score_table import score_table


@tf.function(experimental_relax_shapes=True)
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        _, logits, transition_params = model(x, training=True)
        loss = crf_loss(inputs=logits,
                        tag_indices=y,
                        transition_params=transition_params)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def batch_generator(x, y, batch_size):
    n_batch = math.ceil(len(x) / batch_size)
    for i in range(n_batch):
        yield x[i * batch_size:(i + 1) *
                batch_size], y[i * batch_size:(i + 1) * batch_size]


class Tagger(object):
    def __init__(self,
                 encoder_path,
                 optimizer='adam',
                 learning_rate=None,
                 encoder_trainable=True):
        self.encoder_path = encoder_path
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.encoder_trainable = encoder_trainable

        self.model = None
        self.model_bin = None
        self.word_index = None
        self.index_word = None

        self.tto = None
        self.model_optimizer = None

    def evaluate_table(self, x, y):
        preds = self.predict(x)
        return score_table(preds, y)

    def fit(self,
            x,
            y,
            epochs=1,
            batch_size=32,
            shuffle=True,
            validation_data=None):
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        assert len(x[0]) == len(
            y[0]), 'Lengths of each elements in X should as same as Y'
        # Convert x from [str0, str1] to [[str0], [str1]]
        x = [[xx] for xx in x]
        if not self.model:
            logging.info('build model')
            word_index, index_word = get_tags(y)
            self.word_index = word_index
            self.index_word = index_word
            logging.info(f'tags count: {len(word_index)}')
            logging.info('build model')
            self.model = build_model(
                encoder_path=self.encoder_path,
                tag_size=len(word_index),
                index_word=index_word,
                encoder_trainable=self.encoder_trainable)

            logging.info('build optimizer')
            if self.optimizer == 'adamw':
                self.model_optimizer = tfa.optimizers.AdamW(1e-2)
            else:
                if self.learning_rate is not None:
                    self.model_optimizer = tf.keras.optimizers.Adam(
                        lr=self.learning_rate)
                else:
                    self.model_optimizer = tf.keras.optimizers.Adam()

        tto = ToTokens(self.word_index)

        logging.info('start training')

        for i in range(epochs):
            if shuffle:
                x, y = skshuffle(x, y)
            losses = []
            xdata = tf.data.Dataset.from_generator(
                lambda: iter(x),
                output_types=tf.string,
            )
            xdata = xdata.batch(batch_size)
            ydata = tf.data.Dataset.from_generator(lambda: iter(y),
                                                   output_types=tf.string)
            ydata = ydata.map(tto)
            ydata = ydata.map(lambda x: tf.concat(
                [tf.constant([1]), x, tf.constant([2])], axis=0))
            ydata = ydata.padded_batch(batch_size, padded_shapes=[None])
            data = tf.data.Dataset.zip((xdata, ydata)).prefetch(32)

            pbar = tqdm(data,
                        total=math.ceil(len(x) / batch_size))
            pbar.set_description(f'epoch: {i} loss: {0.:.4f}')
            for batch_x, batch_y in pbar:
                loss = train_step(self.model, self.model_optimizer,
                                  batch_x, batch_y)
                loss = loss.numpy()
                losses.append(loss)
                pbar.set_description(f'epoch: {i} loss: {np.mean(losses):.4f}')
            if validation_data is not None:
                print(
                    self.evaluate_table(validation_data[0],
                                        validation_data[1]))
        logging.info('training done')

    def predict(self, x, batch_size=32):
        assert self.model is not None, 'model not fit or load'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        lengths = [len(xx) for xx in x]
        x = [[xx] for xx in x]
        x = tf.constant(x)
        n_batch = math.ceil(len(x) / batch_size)
        ret = []
        pbar = tqdm(range(n_batch), total=n_batch)
        pbar.set_description('predict')
        for i in pbar:
            y, _, _ = self.model(x[i * batch_size:(i + 1) * batch_size])
            y = y.numpy().tolist()
            batch_lengths = lengths[i * batch_size:(i + 1) * batch_size]
            y = [[self.index_word[yyy] for yyy in yy[1:1 + l]]
                 for l, yy in zip(batch_lengths, y)]
            ret += y
        return ret

    def __getstate__(self):
        """pickle serialize."""
        assert self.model is not None, 'model not fit or load'
        if self.model_bin is None:
            with TempDir() as td:
                self.model.save(td, include_optimizer=False)
                self.model_bin = serialize_dir(td)
        return {
            'model_bin': self.model_bin,
            'index_word': self.index_word,
            'word_index': self.word_index
        }

    def __setstate__(self, state):
        """pikle deserialize."""
        assert 'model_bin' in state, 'invalid model state'
        with TempDir() as td:
            deserialize_dir(td, state.get('model_bin'))
            self.model_bin = state.get('model_bin')
            self.model = tf.keras.models.load_model(td)
        self.word_index = state.get('word_index')
        self.index_word = state.get('index_word')
