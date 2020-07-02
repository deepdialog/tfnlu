import math
import tensorflow as tf
from tfnlu.utils.temp_dir import TempDir
from tfnlu.utils.serialize_dir import serialize_dir, deserialize_dir
from tfnlu.utils.logger import logger
from .parser_model import ParserModel
from .get_tags import get_tags
from .pos_get_tags import pos_get_tags


class Parser(object):
    def __init__(self,
                 encoder_path,
                 proj0_size=500,
                 proj1_size=100,
                 hidden_size=400,
                 encoder_trainable=False):
        self.encoder_path = encoder_path
        self.proj0_size = proj0_size
        self.proj1_size = proj1_size
        self.hidden_size = hidden_size
        self.encoder_trainable = encoder_trainable
        self.model = None

    def fit(self,
            x, y0, y1, epochs=1, batch_size=32,
            build_only=False, optimizer=None):
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], (tuple, list)), \
            'Elements of X should be tuple or list'

        if not self.model:
            logger.info('parser.fit build model')
            word_index, index_word = get_tags(y0)
            pos_word_index, pos_index_word = pos_get_tags(y1)
            self.model = ParserModel(
                encoder_path=self.encoder_path,
                tag0_size=len(word_index),
                tag1_size=1,  # binary
                proj0_size=self.proj0_size,
                proj1_size=self.proj1_size,
                hidden_size=self.hidden_size,
                encoder_trainable=self.encoder_trainable,
                word_index=word_index,
                index_word=index_word,
                pos_word_index=pos_word_index,
                pos_index_word=pos_index_word)
            self.model._set_inputs(
                tf.keras.backend.placeholder((None, None), dtype='string'))

        self.model.compile(optimizer=(
                optimizer
                if optimizer is not None
                else tf.keras.optimizers.Adam(1e-4)))

        if build_only:
            return

        logger.info('parser.fit start training')

        def generator():
            total = math.ceil(len(x) / batch_size)
            for i in range(total):
                x_batch = tf.ragged.constant(
                    x[i * batch_size:(i + 1) * batch_size]
                ).to_tensor()
                y0_batch = tf.ragged.constant(
                    y0[i * batch_size:(i + 1) * batch_size]
                ).to_tensor()
                y1_batch = tf.ragged.constant(
                    y1[i * batch_size:(i + 1) * batch_size]
                ).to_tensor()
                yield x_batch, (y0_batch, y1_batch)

        dataset = tf.data.Dataset.from_generator(
            generator,
            (tf.string, (tf.string, tf.string)),
            (
                tf.TensorShape([None, None]),
                (
                    tf.TensorShape([None, None, None]),
                    tf.TensorShape([None, None])
                )
            )
        ).prefetch(2)

        for xb, _ in dataset.take(2):
            self.model.predict_on_batch(xb)

        self.model.fit(dataset, epochs=epochs)

    def predict(self, x, batch_size=32):
        assert self.model is not None, 'model not fit or load'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], (tuple, list)), \
            'Elements of X should be tuple or list'
        y0, y1 = [], []
        total_batch = int((len(x) - 1) / batch_size) + 1
        for i in range(total_batch):
            x_batch = x[i * batch_size:(i + 1) * batch_size]
            x_batch = tf.ragged.constant(x_batch).to_tensor()
            a, b = self.model.predict(x_batch)
            y0 += a.tolist()
            y1 += b.tolist()

        mats = y0
        deps = []
        for mat, xx in zip(mats, x):
            z = [[word.decode('utf-8')
                  for col, word in enumerate(line)][:len(xx)]
                 for line in mat][:len(xx)]
            deps.append(z)

        pos = [
            [yyy.decode('utf-8') for yyy in yy][:len(xx)]
            for yy, xx in zip(y1, x)
        ]

        return deps, pos

    def __getstate__(self):
        """pickle serialize."""
        assert self.model is not None, 'model not fit or load'
        with TempDir() as td:
            self.model.save(td, include_optimizer=False)
            model_bin = serialize_dir(td)
        return {'model_bin': model_bin}

    def __setstate__(self, state):
        """pikle deserialize."""
        assert 'model_bin' in state, 'invalid model state'
        with TempDir() as td:
            deserialize_dir(td, state.get('model_bin'))
            self.model = tf.keras.models.load_model(td)
