import pytest
import numpy as np

from naplib.array_ops import forward_fill

def test_forward_fill_axis0():
  arr = np.nan*np.ones((5,4))
  arr[0,1] = 1
  arr[2,0] = 2
  arr[2,2] = 3
  expected = np.array([[nan,  1., nan, nan],
                       [nan,  1., nan, nan],
                       [ 2.,  1.,  3., nan],
                       [ 2.,  1.,  3., nan],
                       [ 2.,  1.,  3., nan]])
  output = forward_fill(arr, axis=0)
  assert np.allclose(output, expected)

def test_forward_fill_axis1():
  arr = np.nan*np.ones((5,4))
  arr[0,1] = 1
  arr[2,0] = 2
  arr[2,2] = 3
  expected = np.array([[nan,  1.,  1.,  1.],
                       [nan, nan, nan, nan],
                       [ 2.,  2.,  3.,  3.],
                       [nan, nan, nan, nan],
                       [nan, nan, nan, nan]])
  output = forward_fill(arr, axis=1)
  assert np.allclose(output, expected)

def test_forward_fill_1d():
  arr = np.nan*np.ones((6,))
  arr[1] = 1
  arr[3,0] = 2
  expected = np.array([np.nan,1,1,2,2,2])
  output = forward_fill(arr, axis=0)
  assert np.allclose(output, expected)

def test_bad_input():
  arr = np.nan*np.ones((5,4))
  arr[0,1] = 1
  arr[2,0] = 2
  arr[2,2] = 3
  with pytest.raises(ValueError) as err:
    _ = forward_fill(arr, axis=2)
  assert 'Axis must be either 0 or 1 but got 2' in str(err)
    
  
