import pytest
import os

from naplib.io import read_htk

def test_read_htk():
    thisfile = os.path.dirname(__file__)
    fn = f'{thisfile}/a1_test.htk'
    data, fs = read_htk(fn)
    assert data.shape == (10000, 1)
    assert fs == 2400

def test_read_htk_no_file_found():
    with pytest.raises(FileNotFoundError):
        read_htk('no_file.htk')

def test_read_htk_return_codes():
    thisfile = os.path.dirname(__file__)
    fn = f'{thisfile}/a1_test.htk'
    data, fs, type_code, data_type = read_htk(fn, return_codes=True)
    assert type_code == 8971
    assert data_type == 'PLP_D_A_0'
    assert data.shape == (10000, 1)
    assert fs == 2400
