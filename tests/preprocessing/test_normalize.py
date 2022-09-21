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

def test_normalize_global_stats_zscore(data):
    norm_data = normalize(data=data['out'], field='resp', axis=None, method='zscore')
    concat_norm_data = np.concatenate(norm_data, axis=0)
    expected_0 = np.array([1.40835244, 1.4282645 , 0.99741153, 0.49558854])
    expected_2 = np.array([-0.94945525, -1.29235897, -1.04269637, -0.93081657])
    expected_3 = np.array([0.57355204, 0.58380365, 0.58652175, 0.58109495])
    assert np.allclose(np.mean(concat_norm_data, axis=None), 0)
    assert np.allclose(np.std(concat_norm_data, axis=None), 1)
    assert np.allclose(norm_data[0][100,:], expected_0)
    assert np.allclose(norm_data[1][100,:], expected_2)
    assert np.allclose(norm_data[2][100,:], expected_3)

def test_normalize_global_stats_center(data):
    norm_data = normalize(data=data['out'], field='resp', axis=None, method='center')
    concat_norm_data = np.concatenate(norm_data, axis=0)
    expected_0 = np.array([3.09387436, 3.13761724, 2.19111768, 1.08871092])
    expected_2 = np.array([-2.08576714, -2.83905943, -2.29059962, -2.04482161])
    expected_3 = np.array([1.25998145, 1.28250223, 1.28847336, 1.27655174])
    assert np.allclose(np.mean(concat_norm_data, axis=None), 0)
    assert np.allclose(norm_data[0][100,:], expected_0)
    assert np.allclose(norm_data[1][100,:], expected_2)
    assert np.allclose(norm_data[2][100,:], expected_3)

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
