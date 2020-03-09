import os
import uuid
import tempfile
import shutil


class TempDir(object):
    def __init__(self):
        self.random_path = os.path.join(tempfile.gettempdir(),
                                        str(uuid.uuid4()))

    def __enter__(self):
        os.makedirs(self.random_path)
        return self.random_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.random_path)
