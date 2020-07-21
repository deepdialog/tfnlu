
import numpy as np
import tensorflow as tf
from tfnlu.utils.tfnlu_model import TFNLUModel
from tfnlu.utils.logger import logger
from tfnlu.utils.config import MAX_LENGTH, DEFAULT_BATCH_SIZE
from .parser_model import ParserModel
from .get_tags import get_tags
from .pos_get_tags import pos_get_tags


class Parser(TFNLUModel):
    def __init__(self,
                 encoder_path=None,
                 proj0_size=500,
                 proj1_size=100,
                 hidden_size=400,
                 n_layers=3,
                 encoder_trainable=False):

        super(Parser, self).__init__()

        self.encoder_path = encoder_path
        self.proj0_size = proj0_size
        self.proj1_size = proj1_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.encoder_trainable = encoder_trainable
        self.model = None

    def fit(self,
            x, y0, y1, epochs=1, batch_size=DEFAULT_BATCH_SIZE,
            build_only=False, optimizer=None, optimizer_encoder=None):
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
                n_layers=self.n_layers,
                encoder_trainable=self.encoder_trainable,
                word_index=word_index,
                index_word=index_word,
                pos_word_index=pos_word_index,
                pos_index_word=pos_index_word)

        if build_only:
            return

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(1e-4)
        if optimizer_encoder is None:
            optimizer_encoder = tf.keras.optimizers.Adam(1e-5)

        self.model.optimizer_encoder = optimizer_encoder
        self.model.compile(optimizer=optimizer)

        logger.info('parser.fit start training')

        def make_generator(data):
            def _gen():
                for item in data:
                    yield np.asarray(item, dtype=str)
            return _gen

        x_dataset = tf.data.Dataset.from_generator(
            make_generator(x),
            tf.string,
            tf.TensorShape([None, ])
        )

        y0_dataset = tf.data.Dataset.from_generator(
            make_generator(y0),
            tf.string,
            tf.TensorShape([None, None])
        )

        y1_dataset = tf.data.Dataset.from_generator(
            make_generator(y1),
            tf.string,
            tf.TensorShape([None, ])
        )

        bucket_boundaries = list(range(MAX_LENGTH // 10, MAX_LENGTH, 50))
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        x_dataset, y0_dataset, y1_dataset = [
            dataset.apply(
                tf.data.experimental.bucket_by_sequence_length(
                    size_func,
                    bucket_batch_sizes=bucket_batch_sizes,
                    bucket_boundaries=bucket_boundaries
                )
            )
            for dataset, size_func in zip(
                (x_dataset, y0_dataset, y1_dataset),
                (
                    lambda x: tf.size(x),
                    lambda x: tf.shape(x)[0],
                    lambda x: tf.size(x)
                )
            )
        ]

        dataset = tf.data.Dataset.zip((x_dataset, (y0_dataset, y1_dataset)))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        for xb, (y0b, y1b) in dataset.take(2):
            self.model.predict_on_batch(xb)

        self.model.fit(dataset, epochs=epochs)
        dataset = None

    def predict(self, x, batch_size=32):
        assert self.model is not None, 'model not fit or load'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], (tuple, list)), \
            'Elements of X should be tuple or list'
        y0, y1 = [], []
        total_batch = int((len(x) - 1) / batch_size) + 1

        max_length = MAX_LENGTH + 2
        for i in range(total_batch):
            x_batch = x[i * batch_size:(i + 1) * batch_size]
            # 因为输入变量是不定长的字符串list
            # 如果不对齐的话，tensorflow会尝试构造、跟踪多个graph
            x_batch = [
                xx + [''] * (max_length - len(xx))
                for xx in x_batch
            ]
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
