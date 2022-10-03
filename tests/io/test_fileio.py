import os
import pytest
import numpy as np

from naplib.io import save, load, import_data, export_data

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


# import_data tests

def test_import_Data_works():
    thisfile = os.path.dirname(__file__)
    fn = f'{thisfile}/../../naplib/io/sample_data/demo_data.mat'
    data = import_data(fn)
    assert isinstance(data, Data)
    data = import_data(fn, useloadmat=False)
    assert isinstance(data, Data)

def test_import_Data_fields():
    thisfile = os.path.dirname(__file__)
    fn = f'{thisfile}/../../naplib/io/sample_data/demo_data.mat'
    data = import_data(fn)
    assert all(x==y for x, y in zip(data.fields, ['name','sound','soundf','dataf','duration','befaft','resp','aud','script','chname']))
    assert data[1]['name'] == 'stim02'
    assert data[1]['resp'].shape == (5203, 10)
    assert data[1]['aud'].shape == (5203, 128)
    assert np.allclose(data[1]['resp'][:5,2], np.array([0.17288463, 0.27798367, 0.42636263, 0.46837241, 0.42326083]))
    assert data[0]['script'][:5] == 'IT IS'
    assert data[0]['chname'][2] == 'Fz'
    assert np.allclose(data[-1]['befaft'], np.array([1, 1], dtype='uint8'))
    assert np.allclose(data['sound'][0][20000:20005], np.array([-0.04907227, -0.04403687, -0.03979492, -0.03573608, -0.03210449]))

    data = import_data(fn, useloadmat=False)
    assert set(data.fields)==set(['name','sound','soundf','dataf','duration','befaft','resp','aud','script','chname'])
    assert data[1]['name'] == 'stim02'
    assert data[1]['resp'].shape == (5203, 10)
    assert data[1]['aud'].shape == (5203, 128)
    assert np.allclose(data[1]['resp'][:5,2], np.array([0.17288463, 0.27798367, 0.42636263, 0.46837241, 0.42326083]))
    assert data[0]['script'][:5] == 'IT IS'
    assert data[0]['chname'][2] == 'Fz'
    assert np.allclose(data[-1]['befaft'], np.array([1, 1], dtype='uint8'))
    assert np.allclose(data['sound'][0][20000:20005], np.array([-0.04907227, -0.04403687, -0.03979492, -0.03573608, -0.03210449]))

# export_data tests

def test_export_bad_format(data):
    with pytest.raises(TypeError):
        export_data('fname_out_test.mat', {'bad': 'data'})

def test_export_bad_format(data):
    with pytest.raises(ValueError):
        export_data('fname_out_test.mat', data, fmt='7.2')

def test_export_Data(data):
    export_data('fname_out_test.mat', data)
    pass
