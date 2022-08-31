import pytest
import numpy as np
import matplotlib.pyplot as plt

from naplib.visualization import hierarchicalclusterplot

@pytest.fixture(scope='module')
def data():
	x = np.array([[-3,-5,-7],[0,1,2.],[0.5,1.4,1.9],[0.1,1.1,2.1],[-0.1,1,2],[-2.6,-3.9,-6.9],[-3.5,-5.5,-6.5]])
	varnames = ['x1','x2','x3']
	return {'x': x, 'var': varnames}

def test_correct_clustering(data)
