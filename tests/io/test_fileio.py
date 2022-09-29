import os
import pytest
import numpy as np

from naplib.io import save, load, import_outstruct, export_outstruct

from naplib import Data

@pytest.fixture(scope='module')
def data():
    return Data({'name': ['1','2'],
                 'sound': [np.random.rand(100,), np.random.rand(50,)],
                 'aud': [np.random.rand(100,5), np.random.rand(50,5)],
                 'resp': [np.random.rand(100,2), np.random.rand(50,2)],
                 'other': [np.random.rand(100,3,4), np.random.rand(50,3,4)],
                 'list': [[1,2],[2.4,3.5]],
                 'int': [4,-100]
                 })


# import_outstruct tests

def test_save_dict():
    data = {'x': np.array([1,2,3]), 'string': 'message'}
    save('saved_test_data.pkl', data)

def test_load_dict():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(curr_dir, 'saved_test_data.pkl')
    data = load(fname)
    assert isinstance(data, dict)
    assert np.array_equal(data['x'], np.array([1,2,3]))
    assert data['string'] == 'message'

def test_load_no_file():
    with pytest.raises(FileNotFoundError):
        load('no_file_with_this_name.pkl')

# export_outstruct tests

def test_export_bad_format(data):
    with pytest.raises(TypeError):
        export_outstruct('fname_out_test.mat', {'bad': 'data'})

def test_export_bad_format(data):
    with pytest.raises(ValueError):
        export_outstruct('fname_out_test.mat', data, file_format='7.2')

def test_export_Data(data):
    export_outstruct('fname_out_test.mat', data)
    pass


