import numpy as np
from naplib.preprocessing import rereference, make_contact_rereference_arr
from naplib import Data


def test_create_contact_rereference_arr():
    expected = np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
    g = ['LT1','LT2','RT1','RT2']
    arr = make_contact_rereference_arr(g)
    assert np.allclose(expected, arr)

def test_rereference_avg():
    arr = np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]])
    rng = np.random.default_rng(1)
    data_mat = rng.normal(size=(200,4))
    data = Data({'resp': [data_mat[:120,:], data_mat[120:,:]]})

    data_r, ref = rereference(arr, data, field='resp', method='avg', return_reference=True)
    assert np.allclose(ref[0][:,-1], data_mat[:120,2:].mean(-1))
    assert np.allclose(ref[1][:,1], data_mat[120:,:2].mean(-1))

    assert np.allclose(data_r[0][:,0], data_mat[:120,0] - ref[0][:,0])

