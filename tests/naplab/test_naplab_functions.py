import os
import pytest

from naplib.io import load
from naplib.naplab import align_stimulus_to_recording

@pytest.fixture(scope='module')
def test_align_stimulus_to_recording():
    # Load alignment data
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(curr_dir, 'alignment_test_data.pkl')

    [rec_audio, rec_fs, stim_dict, stim_order] = load(fname)

    alignment_inds, alignment_corrs = align_stimulus_to_recording(rec_audio, rec_fs, stim_dict, stim_order)

    true_trial_inds = [(493978, 1067708),
                      (1134743, 1684059),
                      (1750597, 2263292),
                      (2329729, 2879045),
                      (2945468, 3470370),
                      (3536243, 4073352),
                      (4138888, 4675997),
                      (4741919, 5315649),
                      (5381596, 5930912)]
    true_trial_corrs = [0.7522138260370634,
                        0.7889811677362615,
                        0.7905734388402237,
                        0.7822229391603028,
                        0.7578632507903864,
                        0.7425820566687527,
                        0.7654542295851806,
                        0.7611161884897905,
                        0.808835234596089]
                        
    assert alignment_inds == true_trial_inds
    assert alignment_corrs == true_trial_corrs
