import pytest
import numpy as np
import matplotlib.pyplot as plt

from naplib.visualization import hierarchicalclusterplot

@pytest.fixture(scope='module')
def data():
    x = np.array([[-3,-5,-7],[0,1,2.],[0.5,1.4,1.9],[0.1,1.1,2.1],[-0.1,1,2],[-2.6,-3.9,-6.9],[-3.5,-5.5,-6.5]])
    varnames = ['x1','x2','x3']
    return {'x': x, 'var': varnames}

def test_correct_clustering(data):
    dend, labels, fig, axes = hierarchicalclusterplot(data['x'], n_clusters=2)
    assert np.array_equal(labels, np.array([0,1,1,1,1,0,0])) or np.array_equal(labels, np.array([1,0,0,0,0,1,1]))

def test_varname_labels(data):
    dend, labels, fig, axes = hierarchicalclusterplot(data['x'], varnames=data['var'], n_clusters=2)
    ylbl = axes[1].get_yticklabels()
    assert ylbl[0].get_position()==(0,0)
    assert ylbl[0].get_text()==data['var'][0]
    assert ylbl[1].get_position()==(0,1)
    assert ylbl[1].get_text()==data['var'][1]
    assert ylbl[2].get_position()==(0,2)
    assert ylbl[2].get_text()==data['var'][2]

def test_correct_clustering_othercmap(data):
    dend, labels, fig, axes = hierarchicalclusterplot(data['x'], cmap='gray', n_clusters=2)
    assert np.array_equal(labels, np.array([0,1,1,1,1,0,0])) or np.array_equal(labels, np.array([1,0,0,0,0,1,1]))

