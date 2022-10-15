import pytest
import numpy as np
import copy

from naplib import Data, join_fields, concat

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


# concat tests

def test_concat_axis0_nonshared_fields():
    d1 = Data({'name': ['t1','t2'], 'resp': [[1,2],[3,4,5]], 'extra': ['ex1','ex2']})
    d2 = Data({'name': ['t3','t4'], 'resp': [[6,7],[9,10]], 'extra': ['ex3','ex4']})
    d3 = Data({'name': ['t5','t6'], 'resp': [[11,12,13],[14,15]]})

    d_concat = concat((d1,d2,d3))
    assert len(d_concat) == 6
    assert d_concat['name'] == ['t1', 't2', 't3', 't4', 't5', 't6']
    assert d_concat['resp'] == [[1, 2], [3, 4, 5], [6, 7], [9, 10], [11, 12, 13], [14, 15]]
    assert d_concat.fields == ['name', 'resp']

def test_concat_axis0_all_fields_matching():
    d1 = Data({'name': ['t1','t2'], 'resp': [[1,2],[3,4,5]], 'extra': ['ex1','ex2']})
    d2 = Data({'name': ['t3','t4'], 'resp': [[6,7],[9,10]], 'extra': ['ex3','ex4']})

    d_concat = concat((d1,d2))
    assert len(d_concat) == 4
    assert d_concat['name'] == ['t1', 't2', 't3', 't4']
    assert d_concat['resp'] == [[1, 2], [3, 4, 5], [6, 7], [9, 10]]
    assert d_concat['extra'] == ['ex1', 'ex2', 'ex3', 'ex4']
    assert d_concat.fields == ['name', 'resp', 'extra']

def test_concat_axis1_someshared_fields():
    d3 = Data({'name': ['t1-1','t2-1'], 'resp': [[1,2],[3,4,5]]})
    d4 = Data({'name': ['t1-2','t2-2'], 'meta_data': ['meta1', 'meta2']})
    d_concat = concat((d3, d4), axis=1)
    assert d_concat.fields == ['name', 'resp', 'meta_data']
    assert len(d_concat) == 2
    assert d_concat['name'] == ['t1-1', 't2-1']
    assert d_concat['resp'] == [[1,2],[3,4,5]]
    assert d_concat['meta_data'] == ['meta1', 'meta2']

def test_concat_single_Data_object():
    d3 = Data({'name': ['t1-1','t2-1'], 'resp': [[1,2],[3,4,5]]})
    d_concat = concat([d3], axis=1)
    assert d_concat.fields == ['name', 'resp']

def test_concat_not_data_error():
    with pytest.raises(TypeError) as excinfo:
        d_concat = concat(({'resp': [0,1]}, [2,3]))
    assert 'must be a Data instance' in str(excinfo.value)

def test_concat_axis1_not_all_same_length():
    d3 = Data({'name': ['t1-1'], 'resp': [[1,2]]})
    d4 = Data({'name': ['t1-2','t2-2'], 'meta_data': ['meta1', 'meta2']})
    with pytest.raises(ValueError) as excinfo:
        d_concat = concat((d3, d4), axis=1)
    assert 'All Data objects must be same length' in str(excinfo.value)

def test_concat_axis_not_0_or_1():
    d3 = Data({'name': ['t1-1','t2-1'], 'resp': [[1,2],[3,4,5]]})
    d4 = Data({'name': ['t1-2','t2-2'], 'meta_data': ['meta1', 'meta2']})
    with pytest.raises(ValueError) as excinfo:
        d_concat = concat((d3, d4), axis=2)
    assert 'axis must be 0 or 1' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        d_concat = concat((d3, d4), axis=None)
    assert 'axis must be 0 or 1' in str(excinfo.value)

# join_fields tests

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

