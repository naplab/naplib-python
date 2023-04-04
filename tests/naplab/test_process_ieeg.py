import os
import numpy as np
import pytest

from naplib.io import load
from naplib.naplab import process_ieeg
import matplotlib.pyplot as plt


@pytest.fixture(scope='module')
def small_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'test_processing_pipeline_data_small')
    d = load(os.path.join(dir_path, 'data_small.pkl'))
    return {'path': dir_path, 'data_dict': d}


def test_single_stimuli_pipeline(small_data):

    dir_path = small_data['path']
    true_data = small_data['data_dict']

    data_out = process_ieeg(
        dir_path,
        dir_path,
        bands=['raw', 'theta'],
        phase_amp='both',
        intermediate_fs=100,
        final_fs=100,
        store_all_wav=True,
        store_sounds=True,
        store_spectrograms=False,
        log_level='CRITICAL',
        befaft=[1,1]
    )

    # check trial names based on wav files
    assert data_out['name'] == ['trig_1.wav']

    # check alignment
    assert np.allclose(data_out['alignment_start'][0], 20)
    assert np.allclose(data_out['alignment_end'][0], 30)

    assert data_out['raw'][0].shape == (1200,2) # 2 seconds from befaft, plus 10 seconds of trial at 100 Hz

    # check extracted data (the first 5 seconds of the trial (after befaft) should be the same between the two channels)
    assert np.allclose(data_out['raw'][0][100:600,0], data_out['raw'][0][100:600,1])
    assert not np.allclose(data_out['raw'][0][100:601,0], data_out['raw'][0][100:601,1])

    # check wav channels
    labels_wav = true_data['labels_wav']
    print(data_out[0][labels_wav[0]][15:30])
    print(true_data['wav'][15:30,0])
    assert np.allclose(data_out[0][labels_wav[0]], true_data['wav'][:,0])
    assert np.allclose(data_out[0][labels_wav[1]], true_data['wav'][:,1])
    assert np.allclose(data_out[0][labels_wav[2]], true_data['wav'][:,2])
    assert data_out[0]['wavf'] == 100

    # check stimuli
    print(data_out.fields)
    assert np.allclose(data_out[0]['sound trig_1.wav'], true_data['wav'][:,0])

