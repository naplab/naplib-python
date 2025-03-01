import numpy as np
from naplib.localization import mni152_to_fsaverage, fsaverage_to_mni152

def test_mni152_fsaverage_conversions():
  coords_tmp = np.array([[13.987, 36.5, 10.067], [-10.54, 24.5, 15.555]])
  coords_tmp2 = mni152_to_fsaverage(coords_tmp)
  expected_2 = np.array(
    [[ 14.15310394,  34.73510469,   9.41733728],
    [-10.60987978,  23.11927315,  14.49010227]]
  )
  assert np.allclose(coords_tmp2, expected_2)

  coords_tmp3 = fsaverage_to_mni152(coords_tmp2)
  assert np.allclose(coords_tmp3, coords_tmp)
