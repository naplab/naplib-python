import pytest
import numpy as np
import copy

from naplib import Data, join_fields

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
    return {'out': Data(data_tmp), 'x': [x,x2,x3]}

def test_bracket_indexing(data):
    outstruct = data['out']
    assert np.array_equal(outstruct[-1]['resp'], data['x'][-1])
    assert np.array_equal(outstruct['resp'][-1], data['x'][-1])

def test_create_outstruct_from_list():
    data = [{'x': np.array([1,2]), 'y': 'y0'}, {'x': np.array([3,4]), 'y': 'y1'}]
    outstruct = Data(data)
    assert np.array_equal(outstruct['x'][0], np.array([1,2]))
    assert np.array_equal(outstruct['x'][1], np.array([3,4]))
    assert outstruct['y'][0] == 'y0'
    assert outstruct['y'][1] == 'y1'

def test_create_outstruct_from_dict():
    data = {'x': [np.array([1,2]), np.array([3,4])], 'y': ['y0', 'y1']}
    outstruct = Data(data)
    assert np.array_equal(outstruct['x'][0], np.array([1,2]))
    assert np.array_equal(outstruct['x'][1], np.array([3,4]))
    assert outstruct['y'][0] == 'y0'
    assert outstruct['y'][1] == 'y1'

def test_create_outstruct_from_dict_different_lengths():
    data = {'x': [np.array([1,2]), np.array([3,4])], 'y': ['y0', 'y1', 'y2']}
    with pytest.raises(ValueError):
        outstruct = Data(data)

def test_create_outstruct_from_dict_not_lists_inside():
    data = {'x': [np.array([1,2]), np.array([3,4])], 'y': 'not a list'}
    with pytest.raises(TypeError):
        outstruct = Data(data)

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
    expected = "Data object of 3 trials containing 6 fields\n[{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}\n{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}\n{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}]\n"
    assert expected == tmp

def test_get_string_representation_2trials(data):
    tmp = data['out'][:2].__repr__()
    expected = "Data object of 2 trials containing 6 fields\n[{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}\n{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}]\n"
    assert expected == tmp

def test_get_string_representation_4trials(data):
    out_tmp = copy.copy(data['out'])
    out_tmp.append(data['out'][-1])
    print(len(out_tmp))
    tmp = out_tmp.__repr__()
    print(tmp)
    expected = "Data object of 4 trials containing 6 fields\n[{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}\n{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}\n\n...\n{\"name\": <class 'int'>, \"sound\": <class 'int'>, \"soundf\": <class 'int'>, \"resp\": <class 'numpy.ndarray'>, \"dataf\": <class 'int'>, \"befaft\": <class 'numpy.ndarray'>}]\n"
    print(len(tmp))
    print(len(expected))
    assert expected == tmp

def test_slicing_outstruct(data):
    outstruct = data['out']
    for trial, x in zip(outstruct[1:], data['x'][1:]):
        assert np.array_equal(trial['resp'], x)
        assert trial['dataf'] == 100

def test_join_fields_axis0(data):
    outstruct = data['out']
    second_out = copy.copy(outstruct)
    joined_resp = join_fields([outstruct, second_out], fieldname='resp', axis=0)
    assert np.array_equal(joined_resp[0], np.concatenate([data['x'][0], data['x'][0]], axis=0))
    assert np.array_equal(joined_resp[1], np.concatenate([data['x'][1], data['x'][1]], axis=0))
    assert np.array_equal(joined_resp[2], np.concatenate([data['x'][2], data['x'][2]], axis=0))

def test_join_fields_axis1(data):
    outstruct = data['out']
    second_out = copy.copy(outstruct)
    joined_resp = join_fields([outstruct, second_out], fieldname='resp', axis=1)
    assert np.array_equal(joined_resp[0], np.concatenate([data['x'][0], data['x'][0]], axis=1))
    assert np.array_equal(joined_resp[1], np.concatenate([data['x'][1], data['x'][1]], axis=1))
    assert np.array_equal(joined_resp[2], np.concatenate([data['x'][2], data['x'][2]], axis=1))

def test_join_fields_not_outstruct(data):
    outstruct = data['out']
    with pytest.raises(TypeError):
        joined_resp = join_fields([outstruct, data['x']], fieldname='resp', axis=0)

def test_join_fields_not_nparray(data):
    outstruct = data['out']
    second_out = copy.copy(outstruct)
    with pytest.raises(TypeError):
        joined_resp = join_fields([outstruct, second_out], fieldname='name', axis=0)

