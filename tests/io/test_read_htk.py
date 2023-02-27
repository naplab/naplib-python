import pytest
import numpy as np

from naplib.io import read_htk

def test_read_htk():
    data, fs = read_htk('out_test.htk')
    assert data.shape == (10000, 1)
    assert fs == 2400

def test_read_htk_no_file_found():
    with pytest.raises(FileNotFoundError) as e:
        read_htk('no_file.htk')

def test_read_htk_return_codes():
    data, fs, type_code, data_type = read_htk('out_test.htk', return_codes=True)
    assert type_code = 8971
    assert data_type == 'PLP_D_A_0'
    assert data.shape == (10000, 1)
    assert fs == 2400
