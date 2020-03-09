import tensorflow as tf
import tensorflow_addons as tfa

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


class Tagger(object):
    def __init__(self, encoder_path, optimizer='adamw'):
        self.encoder_path = encoder_path
        self.optimizer = optimizer
        self.predict_model = None
        self.model_bin = None

    def fit(self, x, y):
        assert self.predict_model is None, 'You cannot train this model again'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        assert len(x[0]) == len(
            y[0]), 'Lengths of each elements in X should as same as Y'
        # Convert x from [str0, str1] to [[str0], [str1]]
        x = [[xx] for xx in x]
        word_index, index_word = get_tags(y)
        print(f'tags count: {len(word_index)}')
        print('build model')
        train_model, predict_model = build_model(
            encoder_path=self.encoder_path,
            tag_size=len(word_index),
            index_word=index_word)
        tto = ToTokens(word_index)
        print('build optimizers')
        if self.optimizer == 'adamw':
            optimizer = tfa.optimizers.AdamW(1e-2)
        else:
            optimizer = tf.keras.optimizers.Adam()
        print('start training')

        for i in range(3):
            loss = train_step(train_model, optimizer, tf.constant(x),
                              tto(tf.ragged.constant(y).to_tensor()))
            loss = loss.numpy()
            print(f'{i} loss: {loss:.4f}')
        print('training done')
        self.predict_model = predict_model

    def predict(self, x):
        assert self.predict_model is not None, 'model not fit or load'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        x = [[xx] for xx in x]
        y = self.predict_model(tf.constant(x))
        return y

    def __getstate__(self):
        """pickle serialize."""
        assert self.predict_model is not None, 'model not fit or load'
        if self.model_bin is None:
            with TempDir() as td:
                self.predict_model.save(td, include_optimizer=False)
                self.model_bin = serialize_dir(td)
        return {'model_bin': self.model_bin}

    def __setstate__(self, state):
        """pikle deserialize."""
        assert 'model_bin' in state, 'invalid model state'
        with TempDir() as td:
            deserialize_dir(td, state.get('model_bin'))
            self.model_bin = state.get('model_bin')
            self.predict_model = tf.keras.models.load_model(td)
