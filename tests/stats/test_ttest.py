import pytest
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp

from naplib.stats import ttest

@pytest.fixture(scope='module')
def data():
    data = [[1, 15, 3, 178.7708],[2, 16, 6, 168.4660],[3, 17, 5, 169.9513],[4, 18, 7, 162.0778],[5, 19, 4, 170.1884],[6, 20, 8, 156.9287],[7, 21, 1, 175.4092],[8, 22, 2, 173.3972],[9, 23, 7, 154.4907],[10, 24, 5, 158.3642],[11, 25, 1, 172.1033],[12, 26, 3, 162.6648],[13, 27, 2, 165.4449],[14, 28, 8, 142.2121],[15, 29, 4, 154.3557],[16, 30, 6, 145.6544],[17, 31, 3, 155.5286],[18, 32, 4, 150.5144],[19, 33, 7, 137.8262],[20, 34, 1, 160.1183],[21, 35, 2, 155.4419],[22, 36, 8, 127.1715],[23, 37, 5, 138.0237],[24, 38, 6, 133.4589],[25, 39, 4, 139.3813],[26, 40, 3, 145.1997],[27, 41, 7, 123.7259],[28, 42, 5, 130.7300],[29, 43, 8, 114.1148],[30, 44, 1, 151.1943],[31, 45, 6, 121.7235],[32, 46, 2, 140.9424]]
    data = pd.DataFrame([[1, 15, 3, 178.7708],[2, 16, 6, 168.4660],[3, 17, 5, 169.9513],[4, 18, 7, 162.0778],[5, 19, 4, 170.1884],[6, 20, 8, 156.9287],[7, 21, 1, 175.4092],[8, 22, 2, 173.3972],[9, 23, 7, 154.4907],[10, 24, 5, 158.3642],[11, 25, 1, 172.1033],[12, 26, 3, 162.6648],[13, 27, 2, 165.4449],[14, 28, 8, 142.2121],[15, 29, 4, 154.3557],[16, 30, 6, 145.6544],[17, 31, 3, 155.5286],[18, 32, 4, 150.5144],[19, 33, 7, 137.8262],[20, 34, 1, 160.1183],[21, 35, 2, 155.4419],[22, 36, 8, 127.1715],[23, 37, 5, 138.0237],[24, 38, 6, 133.4589],[25, 39, 4, 139.3813],[26, 40, 3, 145.1997],[27, 41, 7, 123.7259],[28, 42, 5, 130.7300],[29, 43, 8, 114.1148],[30, 44, 1, 151.1943],[31, 45, 6, 121.7235],[32, 46, 2, 140.9424]])
    data.columns = ['Participant','Age','Alcohol','Number']
    data['Age'] = data['Age'].astype('float')
    data['old'] = data['Age'] >= 31
    data['older'] = data['Age'] >= 40
    return data

@pytest.fixture(scope='module')
def data2():
    rng = np.random.default_rng(10)
    x1 = rng.normal(size=(10,1))
    x2 = rng.normal(size=(10,1))+4
    x = np.concatenate([x1,x2])
    x1_group = np.zeros((10,))
    x2_group = np.ones((10,))
    groups = np.concatenate([x1_group,x2_group])
    behavior = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
    return {'x': x, 'x1': x1, 'x2': x2, 'groups': groups, 'behavior': behavior}

def test_relative_ttest(data):
    tval, pval = ttest(data['Number'].values[data['old']],data['Number'].values[~data['old']])
    tval_expected, pval_expected = ttest_rel(data['Number'].values[data['old']],data['Number'].values[~data['old']])
    assert np.allclose(tval, tval_expected, rtol=1e-8, atol=1e-12)
    assert np.allclose(pval, pval_expected, rtol=1e-8, atol=1e-12)

def test_independent_ttest(data):
    tval, pval = ttest(data['Number'].values, classes=data['old'])
    tval_expected, pval_expected = ttest_ind(data['Number'].values[data['old']],data['Number'].values[~data['old']])
    assert np.allclose(tval, tval_expected, rtol=1e-8, atol=1e-12)
    assert np.allclose(pval, pval_expected, rtol=1e-8, atol=1e-12)

def test_1samp_ttest(data):
    tval, pval = ttest(data['Number'].values)
    tval_expected, pval_expected = ttest_1samp(data['Number'].values, 0)
    assert np.allclose(tval, tval_expected, rtol=1e-8, atol=1e-12)
    assert np.allclose(pval, pval_expected, rtol=1e-8, atol=1e-12)

def test_independent_ttest_with_cat_feat(data):
    tval, pval = ttest(data['Number'].values, classes=data['old'], cat_feats=data[['older']])
    assert np.allclose(tval, -3.9610399459745653)
    assert np.allclose(pval, 0.00044466367925574647, atol=1e-14)

def test_independent_ttest_with_cat_feat_different_from_without(data2):
    x = data2['x']
    groups = data2['groups']
    behavior = data2['behavior']
    x1 = data2['x1']
    x2 = data2['x2']
    tval_1, pval_1 = ttest(x, classes=groups, cat_feats=np.concatenate([behavior, behavior], axis=0))
    tval_2, pval_2 = ttest(x, classes=groups)
    tval_scipy, pval_scipy = ttest_ind(x2,x1)
    assert np.allclose(tval_1, 11.682487719722063)
    assert np.allclose(tval_2, tval_scipy)
    assert not np.allclose(tval_1, tval_scipy)
    assert np.allclose(pval_1, 3.0312825227658624e-09, atol=1e-14)
    assert np.allclose(pval_2, pval_scipy)
    assert not np.allclose(pval_1, pval_scipy, atol=1e-14)

def test_relative_ttest_with_cat_feat_different_from_without(data2):
    x = data2['x']
    groups = data2['groups']
    behavior = data2['behavior']
    x1 = data2['x1']
    x2 = data2['x2']
    tval_1, pval_1 = ttest(x2, x1, cat_feats=behavior)
    tval_2, pval_2 = ttest(x2, x1)
    tval_scipy, pval_scipy = ttest_rel(x2,x1)
    assert np.allclose(tval_1, 11.761977075053782)
    assert np.allclose(tval_2, tval_scipy)
    assert not np.allclose(tval_1, tval_scipy)
    assert np.allclose(pval_1, 7.272560323998328e-06)
    assert np.allclose(pval_2, pval_scipy)
    assert not np.allclose(pval_1, pval_scipy, atol=1e-14)

def test_con_feats_different_from_cat_feats(data2):
    behavior = data2['behavior']
    x1 = data2['x1']
    x2 = data2['x2']
    tval_1, pval_1 = ttest(x2, x1, cat_feats=behavior)
    tval_2, pval_2 = ttest(x2, x1, con_feats=behavior)
    assert np.allclose(tval_1, 11.761977075053782)
    assert np.allclose(pval_1, 7.272560323998328e-06, atol=1e-14)
    assert np.allclose(tval_2, 12.896183622283147)
    assert np.allclose(pval_2, 1.2360781608964193e-06, atol=1e-14)

def test_cat_feats_and_con_feats_1samp_ttest(data2):
    x = data2['x']
    groups = data2['groups']
    behavior = data2['behavior']
    tval, pval = ttest(x, cat_feats=groups, con_feats=np.concatenate([behavior, behavior], axis=0))
    assert np.allclose(tval, -1.374171230793057)
    assert np.allclose(pval, 0.18723611138878798)

def test_cat_feats_and_con_feats_overlap_names_error(data2):
    x = data2['x']
    groups = data2['groups']
    behavior = data2['behavior']
    with pytest.raises(ValueError) as err:
        tval, pval = ttest(x, cat_feats={'groups': groups}, con_feats={'groups': np.concatenate([behavior, behavior], axis=0)})
    assert 'cat_feats and con_feats have overlapping key names' in str(err)

def test_result_same_with_dict_or_array_or_df_cat_feat(data2):
    x = data2['x']
    groups = data2['groups']
    tval, pval = ttest(x, cat_feats={'group': groups})
    tval1, pval1 = ttest(x, cat_feats=pd.DataFrame({'group': groups}))
    tval2, pval2 = ttest(x, cat_feats=groups)
    assert np.allclose(tval, tval1)
    assert np.allclose(tval, tval2)
    assert np.allclose(pval, pval1)
    assert np.allclose(pval, pval2)

def test_result_same_with_dict_or_array_or_df_con_feat(data2):
    x = data2['x']
    groups = data2['groups']
    tval, pval = ttest(x, cat_feats={'group': groups})
    tval1, pval1 = ttest(x, cat_feats=pd.DataFrame({'group': groups}))
    tval2, pval2 = ttest(x, cat_feats=groups)
    assert np.allclose(tval, tval1)
    assert np.allclose(tval, tval2)
    assert np.allclose(pval, pval1, rtol=1e-14)
    assert np.allclose(pval, pval2, rtol=1e-14)

def test_wrong_number_of_args_ttest(data2):
    x = data2['x']
    groups = data2['groups']
    with pytest.raises(ValueError) as err:
        tval, pval = ttest(cat_feats={'group': groups})
    assert 'Must provide either 1 or 2' in str(err)

    with pytest.raises(ValueError) as err:
        tval, pval = ttest(x, x, x, cat_feats={'group': groups})
    assert 'Must provide either 1 or 2' in str(err)

def test_bad_cat_feats_type(data2):
    with pytest.raises(TypeError) as err:
        tval, pval = ttest(data2['x'], cat_feats=1)
    assert 'cat_feats must be a' in str(err)

def test_bad_con_feats_type(data2):
    with pytest.raises(TypeError) as err:
        tval, pval = ttest(data2['x'], con_feats=1)
    assert 'con_feats must be a' in str(err)

def test_classes_not_0_and_1(data2):
    with pytest.raises(ValueError) as err:
        tval, pval = ttest(data2['x'], classes=np.ones_like(data2['x']))
    assert 'classes must be an array of only zeros and ones' in str(err)
    with pytest.raises(ValueError) as err:
        tval, pval = ttest(data2['x'], classes=-1*data2['groups'])
    assert 'classes must be an array of only zeros and ones' in str(err)
    groups = data2['groups']
    groups[-1] = 2
    with pytest.raises(ValueError) as err:
        tval, pval = ttest(data2['x'], classes=groups)
    assert 'classes must be an array of only zeros and ones' in str(err)

def test_x_y_not_arrays(data2):
    with pytest.raises(ValueError) as err:
        tval, pval = ttest(data2['x'], classes=np.ones_like(data2['x']))
    assert 'classes must be an array of only zeros and ones' in str(err)

