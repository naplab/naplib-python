import pytest
import numpy as np

from naplib import Data
from naplib.preprocessing import normalize

@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    x = 5+3*rng.random(size=(800,4))
    x2 = 1+2*rng.random(size=(700,4))
    x3 = 6+0.1*rng.random(size=(400,4))
    data_tmp = []
    for i, xx in enumerate([x,x2,x3]):
        data_tmp.append({'name': i, 'sound': 0, 'soundf': 100, 'resp': xx, 'dataf': 100, 'befaft': np.array([1.,1.])})
    return {'out': Data(data_tmp), 'x': [x,x2,x3]}


def test_normalize_properly_centers_and_scales_zscore(data):
    norm_data = normalize(data=data['out'], field='resp', method='zscore')
    concat_norm_data = np.concatenate(norm_data, axis=0)
    assert np.allclose(concat_norm_data.mean(0), np.array([0,0,0,0]), atol=1e-12)
    assert np.allclose(concat_norm_data.std(0), np.array([1,1,1,1]), atol=1e-12)

def test_normalize_properly_centers_and_scales_center(data):
    norm_data = normalize(data=data['out'], field='resp', method='center')
    concat_norm_data = np.concatenate(norm_data, axis=0)
    assert np.allclose(concat_norm_data.mean(0), np.array([0,0,0,0]), atol=1e-12)
    assert not np.allclose(concat_norm_data.std(0), np.array([1,1,1,1]), atol=1e-12)

def test_normalize_properly_centers_and_scales_zscore_axis1(data):
    outstruct = data['out']
    outstruct['resp'] = [x.T for x in outstruct['resp']]
    norm_data = normalize(data=outstruct, field='resp', axis=1, method='zscore')
    concat_norm_data = np.concatenate(norm_data, axis=1)
    assert np.allclose(concat_norm_data.mean(1), np.array([0,0,0,0]), atol=1e-12)
    assert np.allclose(concat_norm_data.std(1), np.array([1,1,1,1]), atol=1e-12)

def test_normalize_properly_centers_and_scales_zscore_from_list(data):
    norm_data = normalize(field=data['x'], method='zscore')
    concat_norm_data = np.concatenate(norm_data, axis=0)
    assert np.allclose(concat_norm_data.mean(0), np.array([0,0,0,0]), atol=1e-12)
    assert np.allclose(concat_norm_data.std(0), np.array([1,1,1,1]), atol=1e-12)

def test_normalize_properly_centers_and_scales_zscore_from_np_array(data):
    tmp = [x[np.newaxis,:400,:] for x in data['x']]
    data_array = np.concatenate(tmp, axis=0)
    norm_data = normalize(field=data_array, method='zscore', axis=1)
    concat_norm_data = np.concatenate(norm_data, axis=0)
    assert np.allclose(concat_norm_data.mean(0), np.array([0,0,0,0]), atol=1e-12)
    assert np.allclose(concat_norm_data.std(0), np.array([1,1,1,1]), atol=1e-12)

def test_normalize_raises_typeerror_bad_input(data):
    with pytest.raises(TypeError):
        norm_data = normalize(field=(1,2,3,4))
