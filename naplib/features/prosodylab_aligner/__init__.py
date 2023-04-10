import logging
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

_console = logging.StreamHandler()
_console.setLevel(logging.WARNING)

formatter = logging.Formatter("%(levelname)s: %(message)s")
_console.setFormatter(formatter)
logger.addHandler(_console)

def set_logging(level: Union[int, str]):
    if not isinstance(level, (int, str)):
        raise ValueError('Level must be of type int or str')
    
    logger.setLevel(level)
    _console.setLevel(level)


from .aligner import Aligner
from .archive import Archive
from .corpus import Corpus
from .run_aligner import run_aligner
