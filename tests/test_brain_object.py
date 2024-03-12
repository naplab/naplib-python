import pytest
import numpy as np
import copy
import os
import mne
import matplotlib.pyplot as plt

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
    bad_isleft[:4] = False

    correct_dists = data['brain_pial'].distance_from_region(data['coords'], data['isleft'], region='pmHG')
    bad_dists = data['brain_pial'].distance_from_region(data['coords'], bad_isleft, region='pmHG')

    assert np.allclose(correct_dists[4:], bad_dists[4:])
    assert not np.allclose(correct_dists[:4], bad_dists[:4])

def test_compute_dist_from_HG_surf(data):
    dist_from_HG1 = data['brain_pial'].distance_from_region(data['coords'], data['isleft'], region='pmHG', metric='surf')
    expected = np.array([52.67211969, 50.86446306, 46.6258215 , 42.63758415, 39.27460724,
       37.07385395, 36.85639736, 27.13146072, 25.69458138, 49.98035531,
       45.88171027, 42.72107443, 41.56546499, 37.83171644, 35.98377774,
       28.77996954, 32.920761  , 19.91693572, 22.38750331, 36.74425306,
       41.73006064, 30.15477367, 30.96079365, 33.36495598, 43.59780259,
       37.00318716, 34.83448731, 32.22574902, 31.71335103, 30.79269113])

    assert np.allclose(dist_from_HG1, expected, atol=1e-5)

def test_compute_dist_from_HG_euclidean(data):
    dist_from_HG1 = data['brain_pial'].distance_from_region(data['coords'], data['isleft'], region='pmHG', metric='euclidean')
    expected = np.array([51.20138081, 49.58391837, 45.70336713, 41.35918778, 38.044578  ,
       35.23063082, 32.82563014, 25.46049199, 25.09231068, 48.80932856,
       45.00960136, 41.87569965, 40.7384779 , 37.46057727, 34.87385546,
       26.48234884, 25.53328187, 20.43864999, 22.67706513, 32.28229751,
       21.84583143, 26.70976875, 26.61332683, 26.29146341, 20.65065803,
       25.12395657, 24.52664769, 24.98141639, 25.43457889, 26.3676947 ])

    assert np.allclose(dist_from_HG1, expected, atol=1e-5)

def test_annotate_coords(data):
    annots = data['brain_pial'].annotate_coords(data['coords'], data['isleft'])
    expected = np.array(['mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG',
       'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG',
       'pSTG', 'mSTG', 'mSTG', 'mSTG', 'pSTG', 'pSTG', 'pSTG', 'pSTG',
       'pSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG', 'mSTG'], dtype='<U4')

    assert np.array_equal(annots, expected)

def test_remote_tts(data):
    old_labels = data['brain_pial'].label_names
    assert 'TTS' in old_labels
    data['brain_inflated'].remove_tts(method='split')
    assert 'TTS' not in data['brain_inflated'].label_names

def test_split_hg():

    brain1 = Brain('pial', subject_dir='./.fsaverage_tmp/').simplify_labels()
    assert 'pmHG' not in brain1.label_names
    assert 'alHG' not in brain1.label_names

    brain1 = Brain('pial', subject_dir='./.fsaverage_tmp/').split_hg(method='endpoint').simplify_labels()
    assert 'pmHG' in brain1.label_names
    assert 'alHG' in brain1.label_names

    brain1 = Brain('pial', subject_dir='./.fsaverage_tmp/').split_hg(method='six_four').simplify_labels()
    assert 'pmHG' in brain1.label_names
    assert 'alHG' in brain1.label_names
    
    brain1 = Brain('pial', subject_dir='./.fsaverage_tmp/').split_hg(method='median').simplify_labels()
    assert 'pmHG' in brain1.label_names
    assert 'alHG' in brain1.label_names


def test_paint_overlay(data):
    brain1 = Brain('pial', subject_dir='./.fsaverage_tmp/').simplify_labels()
    assert np.array_equal(np.unique(brain1.lh.overlay), np.array([0]))
    assert np.array_equal(np.unique(brain1.rh.overlay), np.array([0]))
    brain1.paint_overlay('STG', value=1)
    assert np.array_equal(np.unique(brain1.lh.overlay), np.array([0,1]))
    assert np.array_equal(np.unique(brain1.rh.overlay), np.array([0,1]))

    # reset back to no overlay
    brain1.reset_overlay()
    assert np.array_equal(np.unique(brain1.lh.overlay), np.array([0]))

def test_mark_overlay(data):
    brain1 = Brain('pial', subject_dir='./.fsaverage_tmp/').simplify_labels()
    t1 = brain1.mark_overlay(np.arange(3), np.array([True, True, False]), inner_radius=10, taper=True)
    assert np.array_equal(np.array([1., 1., 0., 0., 0.]), t1.lh.overlay[:5])
    assert np.array_equal(np.array([0., 0., 1., 0., 0.]), t1.rh.overlay[:5])
    assert np.allclose(t1.lh.overlay.sum(), 4305.875)

    brain2 = Brain('pial', subject_dir='./.fsaverage_tmp/').simplify_labels()
    t2 = brain2.mark_overlay(np.arange(3), np.array([True, True, False]), inner_radius=10, taper=False)
    assert np.array_equal(np.array([1., 1., 0., 0., 0.]), t2.lh.overlay[:5])
    assert np.array_equal(np.array([0., 0., 1., 0., 0.]), t2.rh.overlay[:5])
    assert np.allclose(t2.lh.overlay.sum(), 14314.0)

def test_plot_brain_overlay(data):
    brain1 = Brain('pial', subject_dir='./.fsaverage_tmp/').simplify_labels()
    brain1.paint_overlay('STG', value=1)
    fig, axes = plt.test_plot_brain_overlay()
    assert len(axes) == 2
    plt.close()

def test_mpl_both_hemis(data):

    fig, axes = data['brain_pial'].plot_brain_elecs(data['coords'], data['isleft'], values=np.ones((len(data['coords']),)), hemi='both', backend='mpl')
    assert len(axes) == 3 # includes the colorbar
    plt.close()

    fig, axes = data['brain_pial'].plot_brain_elecs(data['coords'], data['isleft'], hemi='both', backend='mpl')
    assert len(axes) == 2 # no colorbar
    plt.close()


def test_mpl_one_hemi(data):
    colors = np.random.rand(len(data['coords']), 4)
    fig, axes = data['brain_pial'].plot_brain_elecs(data['coords'], data['isleft'], colors=colors, hemi='lh', view='frontal', backend='mpl')
    assert len(axes) == 1
    plt.close()

    fig, axes = data['brain_pial'].plot_brain_elecs(data['coords'], data['isleft'], colors=colors, hemi='rh', backend='mpl')
    assert len(axes) == 1
    plt.close()

def test_plotly_electrode_coloring(data):
    colors = ['k' if isL else 'r' for isL in data['isleft']]
    fig, axes = data['brain_inflated'].plot_brain_elecs(data['coords'], data['isleft'], colors=colors, hemi='both', view='best', backend='plotly')
    assert len(fig.data) == 4
    assert fig.data[0]['x'].shape == (163842,)
    assert fig.data[0]['facecolor'].shape == (327680, 4)

    # check that electrodes were split into hemispheres correctly
    assert 'lh' in fig.data[1]['name']
    assert fig.data[1]['marker']['color'].shape[0] == data['isleft'].sum()
    assert 'rh' in fig.data[3]['name']
    assert fig.data[3]['marker']['color'].shape[0] == (len(data['isleft']) - data['isleft'].sum())

    # check elecs are colored correctly for each hemi
    expected_lh = np.asarray([[0,0,0,255] for _ in range(data['isleft'].sum())]) # all black
    expected_rh = np.asarray([[255,0,0,255] for _ in range(len(data['isleft']) - data['isleft'].sum())]) # all red
    assert np.allclose(expected_lh, fig.data[1]['marker']['color'])
    assert np.allclose(expected_rh, fig.data[3]['marker']['color'])

def test_plotly_electrode_coloring_by_value(data):
    colors = ['k' if isL else 'r' for isL in data['isleft']]
    fig, axes = data['brain_inflated'].plot_brain_elecs(data['coords'], data['isleft'], values=data['isleft'], vmin=-1, vmax=2, cmap='binary', hemi='both', view='medial', backend='plotly')
    assert len(fig.data) == 4
    assert fig.data[0]['x'].shape == (163842,)
    assert fig.data[0]['facecolor'].shape == (327680, 4)

    # check elecs are colored correctly for each hemi
    expected_lh = np.asarray([[85,85,85,255] for _ in range(data['isleft'].sum())]) # all black
    expected_rh = np.asarray([[170,170,170,255] for _ in range(len(data['isleft']) - data['isleft'].sum())]) # all red
    assert np.allclose(expected_lh, fig.data[1]['marker']['color'])
    assert np.allclose(expected_rh, fig.data[3]['marker']['color'])

def test_set_visible(data):
    brain_pial1 = copy.deepcopy(data['brain_pial'])
    starting_visible = brain_pial1.lh.alpha
    assert (starting_visible.sum() == 327680)

    brain_pial1.set_visible('pmHG')
    pmHG_visible_lh = brain_pial1.lh.alpha
    pmHG_visible_rh = brain_pial1.rh.alpha
    print((pmHG_visible_lh.sum(), pmHG_visible_rh.sum()))
    assert (pmHG_visible_lh.sum() == 880)
    assert (pmHG_visible_rh.sum() == 612)

    brain_pial1.set_visible('ITG')
    ITG_visible_lh = brain_pial1.lh.alpha
    ITG_visible_rh = brain_pial1.rh.alpha
    print((ITG_visible_lh.sum(), ITG_visible_rh.sum()))
    assert np.allclose(ITG_visible_lh.sum(), 5072)
    assert np.allclose(ITG_visible_rh.sum(), 4546)

    brain_pial1.set_visible(['pmHG','ITG'])
    ITG_and_pmHG_visible_lh = brain_pial1.lh.alpha
    ITG_and_pmHG_visible_rh = brain_pial1.rh.alpha
    print((ITG_and_pmHG_visible_lh.sum(), ITG_and_pmHG_visible_rh.sum()))

    assert np.allclose(ITG_and_pmHG_visible_lh.sum(), 5952)
    assert np.allclose(ITG_and_pmHG_visible_rh.sum(), 5158)

    brain1.reset_overlay()
    ending_visible = brain_pial1.lh.alpha
    assert (ending_visible.sum() == 327680)







