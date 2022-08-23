import pytest
import numpy as np

from naplib.features import auditory_spectrogram

@pytest.fixture(scope='module')
def small_sound():
    fs = 10000
    t = 5
    f = 800
    samples = np.linspace(0, t, int(fs*t), endpoint=False)
    signal = np.sin(2 * np.pi * f * samples)
    return data

def test_small_sound_linear_factor(small_sound):
    spec = auditory_spectrogram(small_sound, 10000, frame_len=8, tc=4, factor='linear')
    assert spec.shape[1] == 128