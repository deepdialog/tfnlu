import math

import tensorflow as tf
import tensorflow_addons as tfa

from tfnlu.utils.logger import logger
from tfnlu.utils.serialize_dir import serialize_dir, deserialize_dir
from tfnlu.utils.temp_dir import TempDir
from .get_tags import get_tags
from .tagger_model import TaggerModel
from .score_table import score_table
from .check_validation import CheckValidation

MAX_LENGTH = 510
DEFAULT_BATCH_SIZE = 32


class Tagger(object):
    def __init__(self,
                 encoder_path,
                 optimizer='RectifiedAdam',
                 optimizer_arguments={},
                 encoder_trainable=False):
        self.encoder_path = encoder_path
        self.encoder_trainable = encoder_trainable
        self.optimizer = optimizer
        self.optimizer_arguments = optimizer_arguments

        self.model = None
        self.word_index = None
        self.index_word = None

        self.tto = None
        self.model_optimizer = None

    def evaluate_table(self, x, y, batch_size=DEFAULT_BATCH_SIZE):
        preds = self.predict(x, batch_size=batch_size)
        return score_table(preds, y)

    def check_data(self, x, y, batch_size=DEFAULT_BATCH_SIZE):
        if not isinstance(batch_size, int):
            batch_size = DEFAULT_BATCH_SIZE
        assert hasattr(x, '__len__'), 'X should be a array like'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        if y:
            assert len(x) == len(y), 'len(X) must equal to len(y)'
            assert isinstance(y[0], str), 'Elements of y should be string'
            assert len(x[0]) == len(
                y[0].split()
            ), 'Lengths of each elements in X should as same as Y'
        for xx in x:
            if len(xx) > MAX_LENGTH:
                logger.warn(
                    f'Some sample(s) longer than {MAX_LENGTH}, will be cut')
                break
        data = {
            'length': len(x),
            'has_y': y is not None,
            'steps': math.ceil(len(x) / batch_size)
        }
        data['x'] = tf.constant([
            [xx[:MAX_LENGTH]]
            for xx in x
        ])
        if y:
            data['y'] = tf.constant([
                [
                    ' '.join(xx.split()[:MAX_LENGTH])
                ]
                for xx in y
            ])
        return data

    def fit(self,
            x,
            y,
            epochs=1,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            validation_data=None,
            save_best=None):
        data = self.check_data(x, y, batch_size)
        if not self.model:
            logger.info('build model')
            word_index, index_word = get_tags(y)
            logger.info(f'tags count: {len(word_index)}')
            logger.info('build model')
            self.model = TaggerModel(
                encoder_path=self.encoder_path,
                word_index=word_index,
                index_word=index_word,
                encoder_trainable=self.encoder_trainable)
        if not self.model._is_compiled:
            logger.info('build optimizer')
            if self.optimizer.lower() == 'adamw':
                model_optimizer = tf.keras.optimizers.AdamW(
                    **self.optimizer_arguments
                )
            elif self.optimizer.lower() == 'rectifiedadam':
                model_optimizer = tfa.optimizers.RectifiedAdam(
                    **self.optimizer_arguments
                )
            else:
                model_optimizer = tf.keras.optimizers.Adam(
                    **self.optimizer_arguments
                )

            self.model.compile(optimizer=model_optimizer)
        logger.info('check model predict')
        self.model.predict(tf.constant([
            [xx[:MAX_LENGTH]]
            for xx in x[:1]
        ]), verbose=0)
        logger.info('start training')

        self.model.fit(
            data.get('x'),
            data.get('y'),
            epochs=epochs,
            callbacks=[
                CheckValidation(
                    tagger=self,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    save_best=save_best)
            ],
            batch_size=batch_size
        )
        logger.info('training done')

    def predict(self, x, batch_size=DEFAULT_BATCH_SIZE, verbose=1):
        assert self.model is not None, 'model not fit or load'
        data = self.check_data(x, None, batch_size)
        pred = self.model.predict(
            data.get('x'),
            verbose=verbose,
            batch_size=batch_size
        )
        pred = [x.decode('UTF-8') for x in pred.tolist()]
        return pred

    def __getstate__(self):
        """pickle serialize."""
        assert self.model is not None, 'model not fit or load'
        with TempDir() as td:
            self.model.save(td, include_optimizer=False)
            model_bin = serialize_dir(td)
        return {
            'model_bin': model_bin,
            'index_word': self.index_word,
            'word_index': self.word_index,
            'optimizer': self.optimizer,
            'optimizer_arguments': self.optimizer_arguments
        }

    def __setstate__(self, state):
        """pickle deserialize."""
        assert 'model_bin' in state, 'invalid model state'
        with TempDir() as td:
            deserialize_dir(td, state.get('model_bin'))
            self.model = tf.keras.models.load_model(td)
        self.word_index = state.get('word_index')
        self.index_word = state.get('index_word')
        self.optimizer = state.get('optimizer')
        self.optimizer_arguments = state.get('optimizer_arguments')
