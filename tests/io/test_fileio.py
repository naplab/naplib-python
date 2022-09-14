import os
import pytest
import numpy as np

from naplib.io import save, load, import_outstruct

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



