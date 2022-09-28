import pytest

from naplib.io import load_speech_task_data
from naplib import Data as DataClass

def test_load_data():
	data = load_speech_task_data()
	assert isinstance(data, DataClass)
	assert len(data) == 10