import tensorflow as tf
import tensorflow_addons as tfa
from tfnlu.utils.temp_dir import TempDir
from tfnlu.utils.serialize_dir import serialize_dir, deserialize_dir
from .build_model import build_model


@tf.function
def parser_loss(y_true, y_pred):
    mask = tf.cast(tf.math.greater(y_true, 0), tf.int32)
    mask = tf.tile(tf.expand_dims(mask, -1), (1, 1, 1, y_pred.shape[-1]))
    y_pred = y_pred * tf.cast(mask, dtype=tf.float32)
    y_pred = tf.reshape(y_pred, (-1, y_pred.shape[-1]))
    y_true = tf.reshape(y_true, (-1, ))
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = parser_loss(y[0], y_pred[0])
        loss += parser_loss(y[1], y_pred[1])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


class Parser(object):
    def __init__(self,
                 encoder_path,
                 proj0_size,
                 proj1_size,
                 tag0_size,
                 tag1_size,
                 optimizer='adamw'):
        self.encoder_path = encoder_path
        self.optimizer = optimizer
        self.predict_model = None
        self.model_bin = None

    def fit(self, x, y):
        assert self.predict_model is None, 'You cannot train this model again'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        # Convert x from [str0, str1] to [[str0], [str1]]
        x = [[xx] for xx in x]
        print('build model')
        train_model = build_model(encoder_path=self.encoder_path)
        predict_model = train_model
        print('build optimizers')
        if self.optimizer == 'adamw':
            optimizer = tfa.optimizers.AdamW(1e-2)
        else:
            optimizer = tf.keras.optimizers.Adam()
        print('start training')
        for i in range(3):
            loss = train_step(train_model, optimizer, tf.constant(x), y)
            loss = loss.numpy()
            print(f'{i} loss: {loss:.4f}')
        print('training done')
        self.predict_model = predict_model

    def predict(self, x):
        assert self.predict_model is not None, 'model not fit or load'
        assert hasattr(x, '__len__'), 'X should be a list/np.array'
        assert len(x) > 0, 'len(X) should more than 0'
        assert isinstance(x[0], str), 'Elements of X should be string'
        # Convert x from [str0, str1] to [[str0], [str1]]
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
