import pytest
import numpy as np

from naplib.stats import responsive_ttest
from naplib import Data

@pytest.fixture(scope='module')
def outstruct():
    rng = np.random.default_rng(1)
    x = rng.random(size=(800,4))
    x2 = rng.random(size=(700,4))
    x3 = rng.random(size=(400,4))
    x[100:,1] = 10+rng.random(700,)
    x2[100:,1] = 5+rng.random(600,)
    x3[100:,1] = 7+rng.random(300,)
    data_tmp = []
    for xx in [x,x2,x3]:
        data_tmp.append({'resp': xx, 'dataf': 100, 'befaft': np.array([1.,1.])})
    return Data(data_tmp)

def test_responsive_ttest_picks_correct_electrode_from_outstruct(outstruct):
    new_out, stats = responsive_ttest(data=outstruct, resp='resp', sfreq='dataf', befaft='befaft', random_state=2)
    assert new_out[0]['resp'].shape[1] == 1
    assert np.array_equal(stats['significant'], np.array([0,1,0,0]).astype('bool'))
    assert np.allclose(stats['stat'], np.array([ -0.14351689, -43.20687375, 0.08349735, -0.96249784]), atol=1e-7)

def test_responsive_ttest_picks_correct_electrode_pass_args_individually_vs_outstruct_same(outstruct):
    new_resp, stats_resp = responsive_ttest(resp=outstruct['resp'], sfreq=100, befaft=np.array([1.,1.]), random_state=2)
    new_out, stats_out = responsive_ttest(data=outstruct, resp='resp', sfreq='dataf', befaft='befaft', random_state=2)

    assert np.array_equal(new_resp[0], new_out[0]['resp'])
    assert np.array_equal(new_resp[1], new_out[1]['resp'])
    assert np.array_equal(new_resp[2], new_out[2]['resp'])

    assert np.array_equal(stats_resp['stat'], stats_out['stat'])
    assert np.array_equal(stats_resp['pval'], stats_out['pval'])
    assert np.array_equal(stats_resp['significant'], stats_out['significant'])
