import pytest
import numpy as np

from naplib.stats import dprime, fratio

def test_dprime():
    D = np.arange(0, 10, 1/1000).reshape(20,500)
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    all_f, f_stat, f_std = dprime(D, L)
    assert np.allclose(all_f, np.array([0.01785714, 0.00198413, 0.00198413, 0.01785714]), atol=1e-8)
    assert np.allclose(f_stat, 0.009920634920620536, atol=1e-8)
    assert np.allclose(f_std, 0.004582144993561766, atol=1e-8)

def test_fratio_elecmode_all():
    D = np.arange(0, 10, 1/1000).reshape(10,2,500)
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat = fratio(D, L, elec_mode='all')
    assert np.allclose(f_stat, np.array([0.00992063, 0.00992063]), atol=1e-8)

def test_fratio_elecmode_individual():
    D = np.arange(0, 10, 1/1000).reshape(10,2,500)
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat = fratio(D, L, elec_mode='individual')
    truth = np.array([[0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063],
       [0.00992063, 0.00992063]])
    assert np.allclose(f_stat, truth, atol=1e-8)
