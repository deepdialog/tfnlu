
import sys

from tqdm import tqdm
import tensorflow as tf

from tfnlu.utils.logger import logger
from tfnlu.utils.config import MAX_LENGTH, DEFAULT_BATCH_SIZE
from tfnlu.utils.tfnlu_model import TFNLUModel
from .get_tags import get_tags
from .classification_model import ClassificationModel


class Classification(TFNLUModel):
    def __init__(self,
                 encoder_path=None,
                 encoder_trainable=False):

        super(Classification, self).__init__()

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

        def size_xy(x, y):
            return tf.size(x)

        def size_x(x):
            return tf.size(x)

        dataset = tf.data.Dataset.from_generator(**_make_gen(x, y))
        bucket_boundaries = list(range(MAX_LENGTH // 10, MAX_LENGTH, 50))
        dataset = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                size_xy if y is not None else size_x,
                bucket_batch_sizes=[batch_size] * (len(bucket_boundaries) + 1),
                bucket_boundaries=bucket_boundaries
            )
        )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def fit(self,
            x,
            y,
            epochs=1,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            validation_data=None,
            save_best=None,
            optimizer=None,
            optimizer_encoder=None):
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

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(1e-4)

        if optimizer_encoder is None:
            optimizer_encoder = tf.keras.optimizers.Adam(1e-5)

        self.model.optimizer_encoder = optimizer_encoder
        self.model.compile(
            optimizer=optimizer,
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

        pred = []
        total_batch = int((len(x) - 1) / batch_size) + 1
        pbar = range(total_batch)
        if verbose:
            pbar = tqdm(pbar, file=sys.stdout)
        for i in pbar:
            x_batch = x[i * batch_size:(i + 1) * batch_size]
            x_batch = [
                ['[CLS]'] + xx[:MAX_LENGTH] + ['[SEP]']
                for xx in x_batch
            ]
            x_batch = tf.ragged.constant(x_batch).to_tensor()
            p = self.model(x_batch)
            pred += [x.decode('UTF-8') for x in p.numpy().tolist()]

        return pred
