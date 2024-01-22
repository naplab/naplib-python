import pytest
import logging
import numpy as np

from naplib.array_ops import resample_categorical

LOGGER = logging.getLogger(__name__)

@pytest.mark.usefixtures("caplog")
def test_too_few_samples(caplog):
    caplog.set_level(logging.WARNING)
    x = np.array([1,1,1,1,2,2,3,3,4,4,4,4,5,5,5,5])
    new = resample_categorical(x, num=6)
    assert 'New labels are not equivalent to the old labels' in caplog.text
    expected = np.array([1., 1., 3., 4., 5., 5.])
    assert np.allclose(new, expected)

def test_bad_x_shape():
    x = np.ones((12,3,4))
    with pytest.raises(ValueError):
        _ = resample_categorical(x, num=8)

def test_downsample():
    x = np.array([1,1,1,1,2,2,3,3,4,4,4,4,5,5,5,5])
    new = resample_categorical(x, num=8)
    expected = np.array([1., 1., 2., 3., 4., 4., 5., 5.])
    assert np.allclose(new, expected)

def test_upsample():
    x = np.array([1,1,1,1,2,2,3,3,4,4,4,4,5,5,5,5])
    new = resample_categorical(x, num=20)
    expected = np.array([1., 1., 1., 1., 1., 2., 2., 2., 3., 3., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5.])
    assert np.allclose(new, expected)

def test_downsample_2d():
    x = np.reshape([1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4],(12,2))
    new = resample_categorical(x, num=7)
    expected = np.array([[1., 1.],[1., 2.],[2., 3.],[3., 3.],[3., 3.],[4., 4.],[4., 4.]])
    assert np.allclose(new, expected)

def test_upsample_2d():
    x = np.reshape([1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4],(12,2))
    new = resample_categorical(x, num=24)
    expected = np.array([[1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 1.],
       [1., 2.],
       [1., 2.],
       [2., 2.],
       [2., 2.],
       [2., 3.],
       [2., 3.],
       [3., 3.],
       [3., 3.],
       [3., 3.],
       [3., 3.],
       [3., 3.],
       [3., 3.],
       [3., 3.],
       [3., 3.],
       [4., 4.],
       [4., 4.],
       [4., 4.],
       [4., 4.],
       [4., 4.],
       [4., 4.]])
    assert np.allclose(new, expected)
