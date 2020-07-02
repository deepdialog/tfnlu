
import tensorflow as tf
from .serialize_dir import serialize_dir, deserialize_dir
from .temp_dir import TempDir


class TFNLUModel(object):

    def __init__(self):
        self.model = None

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
