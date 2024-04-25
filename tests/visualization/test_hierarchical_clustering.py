import pytest
import numpy as np

from naplib.visualization import hierarchical_cluster_plot

@pytest.fixture(scope='module')
def data():
    x = np.array([
        [-3.0,-5.0,-7.0],
        [ 0.0, 1.0, 2.0],
        [ 0.5, 1.4, 1.9],
        [ 0.1, 1.1, 2.1],
        [-0.1, 1.0, 2.0],
        [-2.6,-3.9,-6.9],
        [-3.5,-5.5,-6.5]
    ])
    varnames = ['x1','x2','x3']
    return {'x': x, 'var': varnames}

def test_correct_clustering(data):
    _, labels, _, _ = hierarchical_cluster_plot(data['x'], n_clusters=2)
    assert np.array_equal(labels, np.array([0,1,1,1,1,0,0])) or np.array_equal(labels, np.array([1,0,0,0,0,1,1]))

def test_correct_coloring(data):
    dend, labels, _, _ = hierarchical_cluster_plot(data['x'], n_clusters=2)
    assert len(set(dend['leaves_color_list'])) == 2
    assert len(np.unique(labels)) == 2

    dend, labels, _, _ = hierarchical_cluster_plot(data['x'], n_clusters=3)
    assert len(set(dend['leaves_color_list'])) == 3
    assert len(np.unique(labels)) == 3

def test_varname_labels(data):
    _, _, _, axes = hierarchical_cluster_plot(data['x'], varnames=data['var'], n_clusters=2)
    ylbl = axes[1].get_yticklabels()
    assert ylbl[0].get_position() == (0,0)
    assert ylbl[0].get_text() == data['var'][0]
    assert ylbl[1].get_position() == (0,1)
    assert ylbl[1].get_text() == data['var'][1]
    assert ylbl[2].get_position() == (0,2)
    assert ylbl[2].get_text() == data['var'][2]

def test_correct_clustering_othercmap(data):
    _, labels, _, _ = hierarchical_cluster_plot(data['x'], cmap='gray', n_clusters=2)
    assert np.array_equal(labels, np.array([0,1,1,1,1,0,0])) or np.array_equal(labels, np.array([1,0,0,0,0,1,1]))
