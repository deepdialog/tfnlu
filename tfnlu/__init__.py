import os
from .parser.parser import Parser
from .tagger.tagger import Tagger
from .classification.classification import Classification
from .utils import serialize_dir, deserialize_dir


VERSION_FILE = os.path.join(os.path.dirname(__file__), 'version.txt')
__version__ = open(VERSION_FILE).read().strip()


__all__ = [
    'Parser', 'Tagger', 'Classification', 'TempDir', 'serialize_dir',
    'deserialize_dir', '__version__'
]
