import numpy as np
import pytest

from naplib.array_ops import center_of_mass

def test_center_of_mass_vanilla():
  tmp = np.arange(100).reshape((10,5,2))
  
  expected = (np.arange(10), 6.29347826*np.ones((5,2)))
  result = center_of_mass(tmp, axis=0)
  assert np.allclose(result[0], expected[0])
  assert np.allclose(result[1], expected[1])

  expected = (np.arange(5), 2.8*np.ones((10,2)))
  result = center_of_mass(tmp, axis=1)
  assert np.allclose(result[0], expected[0])
  assert np.allclose(result[1], expected[1])

  expected = (np.arange(2), 0.66666667*np.ones((10,5)))
  result = center_of_mass(tmp, axis=2)
  assert np.allclose(result[0], expected[0])
  assert np.allclose(result[1], expected[1])

def test_center_of_mass_uneven_x_sampling():
  tmp = np.arange(100).reshape((10,5,2))

  expected = (
    np.array([0, 0.55555556, 1.11111111, 1.66666667, 2.22222222, 2.77777778, 3.33333333, 3.88888889, 4.44444444, 5]),
    np.array([[5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365],
              [5.85341365, 5.85341365]])
  )
  result = center_of_mass(np.array([0,1,2,3,5]), tmp, axis=1, interp_n=10)
  assert np.allclose(result[0], expected[0])
  assert np.allclose(result[1], expected[1])
