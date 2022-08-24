import pytest
import numpy as np

from naplib import OutStruct

@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    x = rng.random(size=(800,4))
    x2 = rng.random(size=(700,4))
    x3 = rng.random(size=(400,4))
    x[100:,1] = 10+rng.random(700,)
    x2[100:,1] = 5+rng.random(600,)
    x3[100:,1] = 7+rng.random(300,)
    data_tmp = []
    for i, xx in enumerate([x,x2,x3]):
        data_tmp.append({'name': i, 'sound': 0, 'soundf': 100, 'resp': xx, 'dataf': 100, 'befaft': np.array([1.,1.])})
    return {'out': OutStruct(data_tmp), 'x': [x,x2,x3]}

def test_bracket_indexing(data):
    outstruct = data['out']
    assert np.array_equal(outstruct[-1]['resp'], data['x'][-1])
    assert np.array_equal(outstruct['resp'][-1], data['x'][-1])

def test_iterating_over_outstruct(data):
    outstruct = data['out']
    for trial, x in zip(outstruct, data['x']):
        assert np.array_equal(trial['resp'], x)
        assert trial['dataf'] == 100

def test_append_trial_with_different_fields_strict(data):
    outstruct = data['out']
    outstruct._strict = True
    # missing some required fields
    new_trial = {'resp': np.random.rand(100,4), 'dataf': 100, 'befaft': np.array([1,1])}
    with pytest.raises(ValueError):
        outstruct.append(new_trial)
    # extra field not in original outstruct
    {'extra_field': 0, 'name': 10, 'sound': 0, 'soundf': 100, 'resp': np.random.rand(100,4), 'dataf': 100, 'befaft': np.array([1.,1.])}
    with pytest.raises(ValueError):
        outstruct.append(new_trial)

def test_get_string_representation(data):
    tmp = data['out'].__repr__()
    expected = "OutStruct of 3 trials containing 6 fields\n[{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}\n{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}\n{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}]\n"
    assert expected == tmp

def test_slicing_outstruct(data):
    outstruct = data['out']
    for trial, x in zip(outstruct[1:], data['x'][1:]):
        assert np.array_equal(trial['resp'], x)
        assert trial['dataf'] == 100