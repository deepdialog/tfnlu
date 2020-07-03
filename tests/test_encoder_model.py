
import os
import uuid
import tempfile
import tensorflow as tf
from tfnlu.utils.encoder_model import UnicodeEncoder


class TestEncoderModel(object):

    def test_encoder_model(self):
        encoder = UnicodeEncoder(embedding_size=4)
        x = tf.ragged.constant([
            list('我爱你'),
            list('讨厌'),
        ]).to_tensor()
        y = encoder(x)
        assert y.shape == (2, 3, 4)
        assert sum(y.numpy()[1][2]) == 0

    def test_encoder_model_save_load(self):
        encoder = UnicodeEncoder(embedding_size=4)
        path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        encoder.save(path)
        encoder2 = tf.saved_model.load(path)
        x = tf.ragged.constant([
            list('我爱你'),
            list('讨厌'),
        ]).to_tensor()
        assert encoder2(x).numpy().sum() == encoder(x).numpy().sum()
