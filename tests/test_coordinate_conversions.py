import numpy as np
import os
import mne
from naplib.localization import mni152_to_fsaverage, fsaverage_to_mni152, src_to_dst

def test_mni152_fsaverage_conversions():
  coords_tmp = np.array([[13.987, 36.
  
  , 10.067], [-10.54, 24.
  
  , 15.555]])
  coords_tmp2 = mni152_to_fsaverage(coords_tmp)
  expected_2 = np.array(
    [[ 14.15310394,  34.73510469,   9.41733728],
    [-10.60987978,  23.11927315,  14.49010227]]
  )
  assert np.allclose(coords_tmp2, expected_2, rtol=1e-3)

  coords_tmp3 = fsaverage_to_mni152(coords_tmp2)
  assert np.allclose(coords_tmp3, coords_tmp, rtol=1e-3)

def test_src_to_dst():
  coords = np.random.rand(2, 3) * 5
  
  os.makedirs('./.fsaverage_tmp', exist_ok=True)
  mne.datasets.fetch_fsaverage('./.fsaverage_tmp/')
  
  src_pial = './.fsaverage_tmp/fsaverage/surf/lh.pial'
  src_sphere = './.fsaverage_tmp/fsaverage/surf/lh.sphere.reg'
  dst_pial = './.fsaverage_tmp/fsaverage/surf/lh.inflated'
  dst_sphere = './.fsaverage_tmp/fsaverage/surf/lh.sphere.reg'

  inflated_coords = src_to_dst(coords, src_pial, src_sphere, dst_pial, dst_sphere)
  
  assert inflated_coords.shape[0] == coords.shape[0]