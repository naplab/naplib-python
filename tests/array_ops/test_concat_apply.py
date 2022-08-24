import pytest
import numpy as np
from scipy.signal import resample

from naplib.array_ops import concat_apply

@pytest.fixture(scope='module')
def data():
    arr_list = []
    arr_list.append(np.arange(0,9).reshape((3,3)))
    arr_list.append(np.arange(10,19).reshape((3,3)))
    return arr_list

def bad_function(arr, axis=0):
    # return an intentionally badly shaped array
    if axis==0:
        return np.pad(arr, [[1,1],[0,0]])
    if axis==1:
        return np.pad(arr, [[0,0],[1,1]])


def test_resample_axis1(data):
    new_data = concat_apply(data, resample, axis=0, function_kwargs={'num':2, 'axis':1})
    expected = [np.array([[0.,2],[3,5],[6,8]]), np.array([[10.,12],[13,15],[16,18]])]
    for new, exp in zip(new_data, expected):
        assert np.array_equal(new, exp)

def test_resample_axis0(data):
    new_data = concat_apply(data, resample, axis=1, function_kwargs={'num':2, 'axis':0})
    expected = [np.array([[0.,1,2],[6,7,8]]), np.array([[10.,11,12],[16,17,18]])]
    for new, exp in zip(new_data, expected):
        assert np.array_equal(new, exp)

def test_bad_function_kwargs_type(data):
    with pytest.raises(TypeError):
        new_data = concat_apply(data, resample, function_kwargs=[2])

def test_bad_function_new_size0(data):
    with pytest.raises(RuntimeError):
        new_data = concat_apply(data, bad_function, axis=0, function_kwargs={'axis':0})

def test_bad_function_new_size1(data):
    with pytest.raises(RuntimeError):
        new_data = concat_apply(data, bad_function, axis=1, function_kwargs={'axis':1})

