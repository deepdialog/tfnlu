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
                 optimizer='adamw'):
        self.encoder_path = encoder_path
        self.proj0_size = proj0_size
        self.proj1_size = proj1_size
        self.optimizer = optimizer
        self.model = None

    def fit(self, x, y0, y1, epochs=1, batch_size=32, build_only=False):
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'

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
                word_index=word_index,
                index_word=index_word,
                pos_word_index=pos_word_index,
                pos_index_word=pos_index_word)

        if not self.model._is_compiled:
            logger.info('parser.fit build optimizers')
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4))

        if build_only:
            return

        # Convert x from [str0, str1] to [[str0], [str1]]
        x = [[xx] for xx in x]
        y1 = [[xx] for xx in y1]

        logger.info('parser.fit start training')

        xiter = tf.data.Dataset.from_generator(
            lambda: iter(x), output_types=tf.string
        ).batch(
            batch_size
        )
        y0iter = tf.data.Dataset.from_generator(
            lambda: iter(y0), output_types=tf.string
        ).map(
            lambda x: tf.RaggedTensor.from_tensor(x)
        ).batch(
            batch_size
        ).map(
            lambda x: x.to_tensor()
        )
        y1iter = tf.data.Dataset.from_generator(
            lambda: iter(y1), output_types=tf.string
        ).batch(
            batch_size
        )

        data_iter = tf.data.Dataset.zip((xiter, (y0iter, y1iter)))

        self.model.fit(data_iter,
                       epochs=epochs,
                       steps_per_epoch=len(x) // batch_size)

    def predict(self, x, batch_size=32):
        assert self.model is not None, 'model not fit or load'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        # Convert x from [str0, str1] to [[str0], [str1]]
        y0, y1 = [], []
        total_batch = math.ceil(len(x) / batch_size)
        for i in range(total_batch):
            a, b = self.model.predict(
                tf.constant([
                    [xx] for xx in x][i*batch_size: (i + 1) * batch_size]))
            y0 += a.tolist()
            y1 += b.tolist()

        mats = y0
        deps = []
        for mat, xx in zip(mats, x):
            z = [[word.decode('utf-8') for word in line][:len(xx) + 2]
                 for line in mat][:len(xx) + 2]
            deps.append(z)

        pos = [
            ' '.join(xx.decode('utf-8').split()[1:-1]) for xx in y1
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
