import logging
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

_console = logging.StreamHandler()
_console.setLevel(logging.WARNING)
logger.addHandler(_console)

def set_logging(level: Union[int, str]):
    '''
    Sets the log level at the submodule level. All functions within this module
    use this log level.

    Parameters
    ----------
    level : string or int
        Any log level that is recognized by python's built-in ``logging`` module
    '''

    if not isinstance(level, (int, str)):
        raise ValueError('Level must be of type int or str')
    
    logger.setLevel(level)
    _console.setLevel(level)


from .aligner import Aligner
from .archive import Archive
from .corpus import Corpus
from .run_aligner import run_aligner
