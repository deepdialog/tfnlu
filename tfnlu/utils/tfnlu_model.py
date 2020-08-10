
import tempfile
import tensorflow as tf
from .serialize_dir import serialize_dir, deserialize_dir


class TFNLUModel(object):

    def __init__(self):
        self.model = None

    def __getstate__(self):
        """pickle serialize."""
        assert self.model is not None, 'model not fit or load'
        with tempfile.TemporaryDirectory() as td:
            if hasattr(self.model, 'optimizer'):
                opt = self.model.optimizer
                self.model.optimizer = tf.keras.optimizers.SGD()
            if hasattr(self.model, 'optimizer_encoder'):
                opt_enc = self.model.optimizer_encoder
                self.model.optimizer_encoder = tf.keras.optimizers.SGD()
            self.model.save(td, include_optimizer=False)
            if hasattr(self.model, 'optimizer'):
                self.model.optimizer = opt
            if hasattr(self.model, 'optimizer_encoder'):
                self.model.optimizer_encoder = opt_enc
            model_bin = serialize_dir(td)
        return {'model_bin': model_bin}

    def __setstate__(self, state):
        """pickle deserialize."""
        assert 'model_bin' in state, 'invalid model state'
        with tempfile.TemporaryDirectory() as td:
            deserialize_dir(td, state.get('model_bin'))
            self.model = tf.keras.models.load_model(td)
