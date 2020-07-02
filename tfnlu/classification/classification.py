
import tensorflow as tf

from tfnlu.utils.logger import logger
from tfnlu.utils.serialize_dir import serialize_dir, deserialize_dir
from tfnlu.utils.temp_dir import TempDir
from .get_tags import get_tags
from .classification_model import ClassificationModel

MAX_LENGTH = 510
DEFAULT_BATCH_SIZE = 32


class Classification(object):
    def __init__(self,
                 encoder_path,
                 encoder_trainable=False):
        self.encoder_path = encoder_path
        self.encoder_trainable = encoder_trainable

        self.model = None
        self.word_index = None
        self.index_word = None

        self.tto = None

    def check_data(self, x, y, batch_size=DEFAULT_BATCH_SIZE):
        if not isinstance(batch_size, int):
            batch_size = DEFAULT_BATCH_SIZE
        assert hasattr(x, '__len__'), 'X should be a array like'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], (tuple, list)), \
            'Elements of X should be tuple or list'
        if y:
            assert len(x) == len(y), 'len(X) must equal to len(y)'
            assert isinstance(y[0], str), 'Elements of y should be string'
        for xx in x:
            if len(xx) > MAX_LENGTH:
                logger.warn(
                    f'Some sample(s) longer than {MAX_LENGTH}, will be cut')
                break

        def _make_gen(x, y=None):
            def _gen_x():
                for xx in x:
                    xx = tf.constant(
                        ['[CLS]'] + xx[:MAX_LENGTH] + ['[SEP]'],
                        tf.string)
                    yield xx

            def _gen_xy():
                for xx, yy in zip(x, y):
                    xx = tf.constant(
                        ['[CLS]'] + xx[:MAX_LENGTH] + ['[SEP]'],
                        tf.string)
                    yy = tf.constant(yy, tf.string)
                    yield xx, yy

            if y is None:
                return {
                    'generator': _gen_x,
                    'output_types': tf.string,
                    'output_shapes': tf.TensorShape([None, ])}
            return {
                    'generator': _gen_xy,
                    'output_types': (tf.string, tf.string),
                    'output_shapes': (
                        tf.TensorShape([None, ]),
                        tf.TensorShape([]))}

        dataset = tf.data.Dataset.from_generator(**_make_gen(x, y))
        dataset = dataset.padded_batch(batch_size).prefetch(2)

        return dataset

    def fit(self,
            x,
            y,
            epochs=1,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            validation_data=None,
            save_best=None,
            optimizer=None):
        data = self.check_data(x, y, batch_size)
        if not self.model:
            logger.info('build model')
            word_index, index_word = get_tags(y)
            logger.info(f'targets count: {len(word_index)}')
            logger.info('build model')
            self.model = ClassificationModel(
                encoder_path=self.encoder_path,
                word_index=word_index,
                index_word=index_word,
                encoder_trainable=self.encoder_trainable)

            self.model._set_inputs(
                tf.keras.backend.placeholder((None, None), dtype='string'))

        self.model.compile(
            optimizer=(
                optimizer
                if optimizer is not None
                else tf.keras.optimizers.Adam(1e-4)),
            metrics=['acc']
        )

        logger.info('check model predict')
        self.model.predict(tf.constant([
            ['[CLS]'] + xx[:MAX_LENGTH] + ['[SEP]']
            for xx in x[:1]
        ]), verbose=0)

        logger.info('start training')
        self.model.fit(data, epochs=epochs)
        logger.info('training done')

    def predict(self, x, batch_size=DEFAULT_BATCH_SIZE, verbose=1):
        assert self.model is not None, 'model not fit or load'
        data = self.check_data(x, None, batch_size)
        pred = self.model.predict(data, verbose=verbose)
        pred = [x.decode('UTF-8') for x in pred.tolist()]
        return pred

    def __getstate__(self):
        """pickle serialize."""
        assert self.model is not None, 'model not fit or load'
        with TempDir() as td:
            self.model.save(td, include_optimizer=False)
            model_bin = serialize_dir(td)
        return {'model_bin': model_bin}

    def __setstate__(self, state):
        """pickle deserialize."""
        assert 'model_bin' in state, 'invalid model state'
        with TempDir() as td:
            deserialize_dir(td, state.get('model_bin'))
            self.model = tf.keras.models.load_model(td)
