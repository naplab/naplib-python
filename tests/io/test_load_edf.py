import os
import pytest
import numpy as np
from scipy.io import loadmat

from naplib.io import load_edf


def test_load_edf():
    curr_dir = os.path.dirname(__file__)
    file = os.path.join(curr_dir, 'tiny_test.edf')
    data = load_edf(file)
    file = os.path.join(curr_dir, 'tiny_test_edf.mat')
    ref = loadmat(file)
    
    assert data['data_f'] == data['wav_f'] == 2048.0
    assert data['data'].shape == (4096, 104)
    assert data['wav'].shape == (4096, 11)
    
    ref_labels = [s.item() for s in ref['header']['label'][0, 0][0]]
    assert all(a == b for a, b in zip(data['labels_data'], ref_labels[:104]))
    assert all(a == b for a, b in zip(data['labels_wav'], ref_labels[104:]))

    assert np.allclose(data['data'], ref['data'])
    assert np.allclose(data['wav'], ref['wav'])


def test_load_edf_no_file_found():
    with pytest.raises(FileNotFoundError):
        load_edf('no_file.edf')


def test_partial_load_edf():
    curr_dir = os.path.dirname(__file__)
    file = os.path.join(curr_dir, 'tiny_test.edf')
    data = load_edf(file, t1=0.5, t2=1.5)
    file = os.path.join(curr_dir, 'tiny_test_edf.mat')
    ref = loadmat(file)

    assert np.allclose(data['data'], ref['data'][1024:3072])
    assert np.allclose(data['wav'], ref['wav'][1024:3072])

