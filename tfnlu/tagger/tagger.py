import math

import tensorflow as tf

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
                 encoder_trainable=False):
        self.encoder_path = encoder_path
        self.encoder_trainable = encoder_trainable

        self.model = None
        self.word_index = None
        self.index_word = None

        self.tto = None

    def evaluate_table(self, x, y, batch_size=DEFAULT_BATCH_SIZE):
        preds = self.predict(x, batch_size=batch_size)
        return score_table(preds, y)

    def check_data(self, x, y, batch_size=DEFAULT_BATCH_SIZE):
        if not isinstance(batch_size, int):
            batch_size = DEFAULT_BATCH_SIZE
        assert hasattr(x, '__len__'), 'X should be a array like'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], (tuple, list)), \
            'Elements of X should be tuple or list'
        if y:
            assert len(x) == len(y), 'len(X) must equal to len(y)'
            assert isinstance(y[0], (tuple, list)), \
                'Elements of y should be tuple or list'
            # 暂时关掉，因为X可能有 [MASK] 等特殊符号
            assert len(x[0]) == len(y[0]), \
                'Lengths of each elements in X should as same as Y'
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
        data['x'] = tf.ragged.constant([
            ['[CLS]'] + xx[:MAX_LENGTH] + ['[SEP]']
            for xx in x
        ]).to_tensor()
        if y:
            data['y'] = tf.ragged.constant([
                ['[CLS]'] + yy[:MAX_LENGTH] + ['[SEP]']
                for yy in y
            ]).to_tensor()
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
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4))

        logger.info('check model predict')
        self.model.predict(data.get('x')[:2], verbose=0)

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
        }

    def __setstate__(self, state):
        """pickle deserialize."""
        assert 'model_bin' in state, 'invalid model state'
        with TempDir() as td:
            deserialize_dir(td, state.get('model_bin'))
            self.model = tf.keras.models.load_model(td)
        self.word_index = state.get('word_index')
        self.index_word = state.get('index_word')
