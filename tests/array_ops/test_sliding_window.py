import pytest
import numpy as np

from naplib.array_ops import sliding_window

@pytest.fixture(scope='module')
def data():
    arr = np.arange(1,5)
    return arr

def test_bad_window_key_idx(data):
    with pytest.raises(ValueError):
        slide = sliding_window(data, 3, window_key_idx=-1)
    with pytest.raises(ValueError):
        slide = sliding_window(data, 3, window_key_idx=4)
    with pytest.raises(ValueError):
        slide = sliding_window(data, 3, window_key_idx=np.nan)
    with pytest.raises(TypeError):
        slide = sliding_window(data, 3, window_key_idx=[1])

def test_slide(data):
    slide = sliding_window(data, 3)
    expected = np.array([[0., 0., 1.],
                         [0., 1., 2.],
                         [1., 2., 3.],
                         [2., 3., 4.]])
    assert np.array_equal(slide, expected)

def test_slide_no_fill_out_of_bounds(data):
    slide = sliding_window(data, 3, fill_out_of_bounds=False)
    expected = np.array([[1., 2., 3.],
                         [2., 3., 4.]])
    assert np.array_equal(slide, expected)


def test_slide_end_window_key_idx(data):
    slide = sliding_window(data, 3, window_key_idx=2)
    expected = np.array([[1., 2., 3.],
                         [2., 3., 4.],
                         [3., 4., 0.],
                         [4., 0., 0.]])
    assert np.array_equal(slide, expected)

def test_slide_middle_window_key_idx(data):
    slide = sliding_window(data, 3, window_key_idx=1)
    expected = np.array([[0., 1., 2.],
                         [1., 2., 3.],
                         [2., 3., 4.],
                         [3., 4., 0.]])
    assert np.array_equal(slide, expected)

def test_slide_middle_window_key_idx_different_fill_value(data):
    slide = sliding_window(data, 3, window_key_idx=1, fill_value=3.3)
    expected = np.array([[3.3, 1., 2.],
                         [1., 2., 3.],
                         [2., 3., 4.],
                         [3., 4., 3.3]])
    assert np.array_equal(slide, expected)
