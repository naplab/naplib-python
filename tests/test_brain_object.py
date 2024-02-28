import pytest
import numpy as np
import copy

import os
import mne


from naplib import Brain

@pytest.fixture(scope='module')
def data():
    coords = np.array([[-47.281147  ,  17.026093  , -21.833099  ],
       [-48.273964  ,  16.155487  , -20.162935  ],
       [-51.101261  ,  13.711058  , -16.258459  ],
       [-55.660889  ,   9.761111  , -12.340655  ],
       [-58.733326  ,   6.046287  ,  -9.626602  ],
       [-60.749279  ,   2.233287  ,  -8.044459  ],
       [-61.26712   ,  -1.939675  ,  -8.582445  ],
       [-63.686226  , -10.447982  ,  -0.445693  ],
       [-63.453224  ,  -9.826311  ,   1.095302  ],
       [-48.792809  ,  15.73144   , -19.34193   ],
       [-51.336754  ,  13.27527   , -15.57861   ],
       [-53.301971  ,  11.016301  , -12.48259   ],
       [-55.044659  ,   9.894337  , -11.228349  ],
       [-57.597462  ,   6.753941  ,  -8.082416  ],
       [-60.594891  ,   2.579503  ,  -6.884331  ],
       [-63.078999  ,  -8.770401  ,  -1.878142  ],
       [-67.419235  , -26.153931  ,  -1.260003  ],
       [-60.28742599, -11.71243477,   5.62593937],
       [-63.12403107, -12.37896156,   4.09772062],
       [ 64.44213867,  -3.16063929,  -6.95104313],
       [ 61.58537674, -23.53317833,  -3.20349312],
       [ 69.31034851, -18.18317795,   1.97798777],
       [ 69.0439682 , -18.64465904,   1.2625511 ],
       [ 68.32962799, -20.90372849,  -0.25190961],
       [ 59.79437256, -23.76178932,  -3.52095652],
       [-56.57900238,  -9.23060513,  -7.33194447],
       [-58.861763  , -11.2859602 ,  -6.18047237],
       [-61.13874054, -11.35863781,  -4.49999475],
       [-60.82435989,  -8.91696072,  -3.20156574],
       [-61.00576019,  -7.45676041,  -3.06485367]])
    isleft = coords[:,0] < 0

    os.makedirs('./.fsaverage_tmp', exist_ok=True)
    mne.datasets.fetch_fsaverage('./.fsaverage_tmp/')

    brain_inflated = Brain('inflated', subject_dir='./.fsaverage_tmp/').split_hg('midpoint').split_stg().simplify_labels()
    brain_pial = Brain('pial', subject_dir='./.fsaverage_tmp/').split_hg('midpoint').split_stg().join_ifg().simplify_labels()
    

    return {'brain_inflated': brain_inflated,
            'brain_pial': brain_pial,
            'coords': coords,
            'isleft': isleft}

def test_bad_isleft_change(data):
    bad_isleft = data['isleft'].copy()
    bad_isleft[:4] = True

    correct_dists = data['brain_pial'].distance_from_region(data['coords'], data['isleft'], region='pmHG')
    bad_dists = data['brain_pial'].distance_from_region(data['coords'], bad_isleft, region='pmHG')

    assert np.allclose(correct_dists[4:], bad_dists[4:])
    assert not np.allclose(correct_dists[:4], bad_dists[:4])

def test_compute_dist_from_HG_surf(data):
    dist_from_HG1 = data['brain_pial'].distance_from_region(data['coords'], data['isleft'], region='pmHG', metric='surf')
    dist_from_HG2 = data['brain_inflated'].distance_from_region(data['coords'], data['isleft'], region='pmHG', metric='surf')
    expected = np.array([52.67211969, 50.86446306, 46.6258215 , 42.63758415, 39.27460724,
       37.07385395, 36.85639736, 27.13146072, 25.69458138, 49.98035531,
       45.88171027, 42.72107443, 41.56546499, 37.83171644, 35.98377774,
       28.77996954, 32.920761  , 19.91693572, 22.38750331, 36.75626353,
       41.13779771, 29.70781128, 30.51372665, 32.71152818, 43.08254325,
       37.00318716, 34.83448731, 32.22574902, 31.71335103, 30.79269113])

    assert np.allclose(dist_from_HG1, expected)
    assert np.allclose(dist_from_HG2, expected)

def test_compute_dist_from_HG_euclidean(data):
    dist_from_HG1 = data['brain_pial'].distance_from_region(data['coords'], data['isleft'], region='pmHG', metric='euclidean')
    dist_from_HG2 = data['brain_inflated'].distance_from_region(data['coords'], data['isleft'], region='pmHG', metric='euclidean')
    expected = np.array([51.17067854, 49.54186483, 45.62676668, 41.22253056, 37.85904187,
       35.00619601, 32.5843648 , 25.10199761, 24.72810672, 48.76114145,
       44.92899899, 41.76565915, 40.60505468, 37.28299408, 34.64679732,
       26.14865343, 25.13892486, 20.05481732, 22.27812812, 32.32367299,
       21.74880021, 26.51097265, 26.42224012, 26.11006209, 20.57905128,
       24.90655226, 24.26172569, 24.67493693, 25.13182362, 26.06956133])

    assert np.allclose(dist_from_HG1, expected)
    assert np.allclose(dist_from_HG2, expected)


def test_plotly_electrode_coloring(data):
    colors = ['k' if isL else 'r' for isL in data['isleft']]
    fig, axes = data['brain_inflated'].plot_brain_elecs(data['coords'], data['isleft'], colors=colors, hemi='both', backend='plotly')
    assert len(fig,data) == 4
    assert fig.data[0]['x'].shape == (163842,)
    assert fig.data[0]['facecolor'].shape == (163842,)

    # check that electrodes were split into hemispheres correctly
    assert 'lh' in fig.data[1]['name']
    assert fig.data[1]['marker']['color'].shape == data['isleft'].sum()
    assert 'rh' in fig.data[3]['name']
    assert fig.data[1]['marker']['color'].shape == (len(data['isleft']) - data['isleft'].sum())

    # check elecs are colored correctly for each hemi
    expected_lh = np.asarray([[0,0,0,255] for _ in range(data['isleft'].sum())]) # all black
    expected_rh = np.asarray([[255,0,0,255] for _ in range(len(data['isleft']) - data['isleft'].sum())]) # all red
    assert np.allclose(expected_lh, fig.data[1]['marker']['color'])
    assert np.allclose(expected_rh, fig.data[3]['marker']['color'])

def test_plotly_electrode_coloring_by_value(data):
    colors = ['k' if isL else 'r' for isL in data['isleft']]
    fig, axes = data['brain_inflated'].plot_brain_elecs(data['coords'], data['isleft'], values=data['isleft'], vmin=-1, vmax=2, cmap='binary', hemi='both', backend='plotly')
    assert len(fig,data) == 4
    assert fig.data[0]['x'].shape == (163842,)
    assert fig.data[0]['facecolor'].shape == (163842,)

    # check elecs are colored correctly for each hemi
    expected_lh = np.asarray([[85,85,85,255] for _ in range(data['isleft'].sum())]) # all black
    expected_rh = np.asarray([[170.170,170,255] for _ in range(len(data['isleft']) - data['isleft'].sum())]) # all red
    assert np.allclose(expected_lh, fig.data[1]['marker']['color'])
    assert np.allclose(expected_rh, fig.data[3]['marker']['color'])







