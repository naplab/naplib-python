import numpy as np

from naplib.stats import discriminability
from naplib.stats.encoding import lda_discriminability

def test_lda_discriminability_single():
    D = np.arange(0, 10, 1/1000).reshape(20,500).T
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    print(D.shape)
    print(L.shape)
    all_f, f_stat, f_std = lda_discriminability(D, L)
    assert np.allclose(all_f, np.array([0.01785714, 0.00198413, 0.00198413, 0.01785714]), atol=1e-8)
    assert np.allclose(f_stat, 0.009920634920620536, atol=1e-8)
    assert np.allclose(f_std, 0.004582144993561766, atol=1e-8)

def test_discriminability_elecmode_all():
    D = np.arange(0, 10, 1/1000).reshape(10,2,500)
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat = discriminability(D, L, elec_mode='all')
    assert np.allclose(f_stat, np.array([0.00992063, 0.00992063]), atol=1e-8)

def test_discriminability_elecmode_all_perm_invariant():
    D = np.arange(0, 10, 1/1000).reshape(10,2,500)
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat1 = discriminability(D, L, elec_mode='all')
    rng = np.random.default_rng(1)
    order = rng.permutation(len(L))
    f_stat2 = discriminability(D[:,:,order], L[order], elec_mode='all')
    assert np.allclose(f_stat1, f_stat2, atol=1e-8)

def test_discriminability_wilks_lambda_elecmode_all_perm_invariant():
    rng = np.random.default_rng(1)
    D = np.arange(0, 10, 1/1000).reshape(10,2,500) + rng.normal(size=(10,2,500))
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat1, pval1 = discriminability(D, L, elec_mode='all', method='wilks-lambda')
    rng = np.random.default_rng(1)
    order = rng.permutation(len(L))
    f_stat2, pval2 = discriminability(D[:,:,order], L[order], elec_mode='all', method='wilks-lambda')
    assert np.allclose(f_stat1, f_stat2, atol=1e-8)
    assert np.allclose(pval1, pval2, atol=1e-8)

def test_discriminability_elecmode_individual():
    D = np.arange(0, 10, 1/1000).reshape(10,2,500)
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat = discriminability(D, L, elec_mode='individual')
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

def test_discriminability_wilks_lambda_elecmode_all():
    rng = np.random.default_rng(1)
    D = np.arange(0, 10, 1/1000).reshape(10,2,500) + rng.normal(size=(10,2,500))
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat, pval = discriminability(D, L, elec_mode='all', method='wilks-lambda')
    assert np.allclose(f_stat, np.array([0.08261644, 0.12792341]), atol=1e-8)
    assert np.allclose(pval, np.array([1., 1.]), atol=1e-8)

def test_discriminability_wilks_lambda_same_as_lda_elecmode_individual():
    rng = np.random.default_rng(1)
    D = np.arange(0, 10, 1/1000).reshape(10,2,500) + rng.normal(size=(10,2,500))
    L = np.concatenate([np.array([1,2,3,4.]) for _ in range(125)], axis=0)
    f_stat1, _ = discriminability(D, L, elec_mode='individual', method='wilks-lambda')
    f_stat2 = discriminability(D, L, elec_mode='individual', method='lda')
    assert np.allclose(f_stat1, f_stat2, atol=1e-8)
