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

@pytest.fixture(scope='module')
def small_data_fs50():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'test_processing_pipeline_data_small_50fs')
    d = load(os.path.join(dir_path, 'data_small_fs50.pkl'))
    return {'path': dir_path, 'data_dict': d}

def test_single_stimuli_pipeline(small_data):

    dir_path = small_data['path']
    true_data = small_data['data_dict']

    data_out = process_ieeg(
        dir_path,
        dir_path,
        stim_dirs={'aud_copy': dir_path},
        bands=['raw', 'theta'],
        phase_amp='both',
        intermediate_fs=100,
        final_fs=100,
        store_all_wav=True,
        store_sounds=True,
        store_spectrograms=True,
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
    assert np.allclose(data_out[0][labels_wav[0]][110:120], np.ones((10,))) # trigger
    assert np.allclose(data_out[0][labels_wav[0]], true_data['wav'][1900:3100,0])
    assert np.allclose(data_out[0][labels_wav[1]], true_data['wav'][1900:3100,1])
    assert np.allclose(data_out[0][labels_wav[2]], true_data['wav'][1900:3100,2])
    assert data_out[0]['wavf'] == 100

    # check stimuli (also should include befaft period now)
    assert np.allclose(data_out[0]['aud_copy sound'], true_data['wav'][1900:3100,0])
    
    # check other bands present
    assert data_out['theta amp'][0].shape == (1200,2)
    assert data_out['theta phase'][0].shape == (1200,2)

    # check spectrograms correct shape
    assert data_out['aud_copy'][0].shape == (1200,128)
    assert np.all(data_out['aud_copy'][0][:100] < 0.01)


def test_single_stimuli_spectrum_inference_method(small_data):

    dir_path = small_data['path']
    true_data = small_data['data_dict']

    data_out = process_ieeg(
        dir_path,
        dir_path,
        bands=['raw', 'theta'],
        phase_amp='both',
        intermediate_fs=100,
        final_fs=100,
        aud_channel_infer_method='spectrum',
        store_all_wav=True,
        store_sounds=True,
        store_spectrograms=True,
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
    assert np.allclose(data_out[0][labels_wav[0]][110:120], np.ones((10,))) # trigger
    assert np.allclose(data_out[0][labels_wav[0]], true_data['wav'][1900:3100,0])
    assert np.allclose(data_out[0][labels_wav[1]], true_data['wav'][1900:3100,1])
    assert np.allclose(data_out[0][labels_wav[2]], true_data['wav'][1900:3100,2])
    assert data_out[0]['wavf'] == 100

    # check stimuli (also should include befaft period now)
    assert np.allclose(data_out[0]['aud sound'], true_data['wav'][1900:3100,0])
    
    # check other bands present
    assert data_out['theta amp'][0].shape == (1200,2)
    assert data_out['theta phase'][0].shape == (1200,2)

    # check spectrograms correct shape
    assert data_out['aud'][0].shape == (1200,128)

def test_single_stimuli_pipeline_with_rereference(small_data):

    dir_path = small_data['path']
    true_data = small_data['data_dict']

    data_out = process_ieeg(
        dir_path,
        dir_path,
        rereference_grid=np.ones((2,2)),
        rereference_method='avg',
        bands=['raw'],
        intermediate_fs=100,
        final_fs=100,
        store_reference=True,
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

    # check extracted data is correctly rereferenced (500 samples that are the same should be 0)
    assert np.allclose(data_out['raw'][0][100:600,0], np.zeros((500,)))

    # check reference is equal to mean of two channels data
    assert np.allclose(data_out[0]['reference'][:,0], true_data['data'][1900:3100].mean(1))


def test_single_stimuli_pipeline_with_rereference_downsample(small_data_fs50):
    """This data has a sampling rate of 50, which is different from wav file stimulus"""
    dir_path = small_data_fs50['path']
    true_data = small_data_fs50['data_dict']

    data_out = process_ieeg(
        dir_path,
        dir_path,
        rereference_grid=np.ones((2,2)),
        rereference_method='pca',
        intermediate_fs = 40,
        final_fs = 10,
        bands=['raw'],
        store_reference=True,
        store_spectrograms=False,
        log_level='CRITICAL',
        befaft=[1,1]
    )

    # check trial names based on wav files
    assert data_out['name'] == ['trig_1.wav']

    # check alignment
    assert np.allclose(data_out['alignment_start'][0], 20)
    assert np.allclose(data_out['alignment_end'][0], 30)

    assert data_out['raw'][0].shape == (120,2) # 2 seconds from befaft, plus 10 seconds of trial at 100 Hz

    # check reference is nan (since two channels are identical)
    assert data_out['reference'][0].shape == (120,2)

    # shape of output correct
    assert data_out['raw'][0].shape == (120,2)

