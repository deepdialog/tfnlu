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


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        _, logits, transition_params = model(x, training=True)
        loss = crf_loss(y, logits, transition_params)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def batch_generator(x, y, batch_size):
    n_batch = math.ceil(len(x) / batch_size)
    for i in range(n_batch):
        yield x[i * batch_size:(i + 1) *
                batch_size], y[i * batch_size:(i + 1) * batch_size]


class Tagger(object):
    def __init__(self, encoder_path, optimizer='adamw'):
        self.encoder_path = encoder_path
        self.optimizer = optimizer
        self.predict_model = None
        self.model_bin = None
        self.word_index = None
        self.index_word = None

    def fit(self, x, y, epochs=1, batch_size=32, shuffle=True):
        assert self.predict_model is None, 'You cannot train this model again'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        assert len(x[0]) == len(
            y[0]), 'Lengths of each elements in X should as same as Y'
        # Convert x from [str0, str1] to [[str0], [str1]]
        x = [[xx] for xx in x]
        word_index, index_word = get_tags(y)
        self.word_index = word_index
        self.index_word = index_word
        logging.info(f'tags count: {len(word_index)}')
        logging.info('build model')
        train_model, predict_model = build_model(
            encoder_path=self.encoder_path,
            tag_size=len(word_index),
            index_word=index_word)
        tto = ToTokens(word_index)
        logging.info('build optimizers')
        if self.optimizer == 'adamw':
            optimizer = tfa.optimizers.AdamW(1e-2)
        else:
            optimizer = tf.keras.optimizers.Adam()
        logging.info('start training')

        for i in range(epochs):
            if shuffle:
                x, y = skshuffle(x, y)
            losses = []
            pbar = tqdm(batch_generator(x, y, batch_size=batch_size),
                        total=math.ceil(len(x) / batch_size))
            pbar.set_description(f'epoch: {i} loss: {0.:.4f}')
            for batch_x, batch_y in pbar:
                loss = train_step(train_model, optimizer, tf.constant(batch_x),
                                  tto(tf.ragged.constant(batch_y).to_tensor()))
                loss = loss.numpy()
                losses.append(loss)
                pbar.set_description(f'epoch: {i} loss: {np.mean(losses):.4f}')
        logging.info('training done')
        self.predict_model = predict_model

    def predict(self, x, batch_size=32):
        assert self.predict_model is not None, 'model not fit or load'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        x = [[xx] for xx in x]
        n_batch = math.ceil(len(x) / batch_size)
        ret = []
        for i in range(n_batch):
            y = self.predict_model(
                tf.constant(x[i * batch_size:(i + 1) * batch_size]))
            y = y.numpy().tolist()
            y = [[self.index_word[yyy] for yyy in yy[:len(xx[0])]]
                 for xx, yy in zip(x, y)]
            ret += y
        return ret

    def __getstate__(self):
        """pickle serialize."""
        assert self.predict_model is not None, 'model not fit or load'
        if self.model_bin is None:
            with TempDir() as td:
                self.predict_model.save(td, include_optimizer=False)
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
            self.predict_model = tf.keras.models.load_model(td)
        self.word_index = state.get('word_index')
        self.index_word = state.get('index_word')
