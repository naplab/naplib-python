import logging
from typing import Union

# Set up logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")
_console.setFormatter(formatter)
_logger.addHandler(_console)

def set_logging(level: Union[int, str]):
    if not isinstance(level, (int, str)):
        raise ValueError('Level must be of type int or str')
    
    _logger.setLevel(level)
    _console.setLevel(level)

def is_logging(level: int) -> bool:
    return _logger.isEnabledFor(level)


import naplib.features
import naplib.segmentation
import naplib.stats
import naplib.visualization
import naplib.encoding
import naplib.io
import naplib.preprocessing
import naplib.array_ops
import naplib.model_selection
import naplib.utils
from .data import Data, join_fields, concat
import naplib.naplab

__version__ = "0.3.1"

