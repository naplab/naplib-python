import os
import numpy as np
import pytest
import scipy.signal
from functools import partial

from naplib.io import load
from naplib.naplab import process_ieeg


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
        stim_order=['trig_1.wav'],
        stim_dirs={'aud_copy': dir_path},
        bands=['raw', 'theta'],
        phase_amp='both',
        intermediate_fs=100,
        final_fs=100,
        store_all_wav=True,
        store_sounds=True,
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
        aud_fn=None,
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


def test_single_stimuli_pipeline_with_custom_rereference(small_data):

    dir_path = small_data['path']
    true_data = small_data['data_dict']

    data_out = process_ieeg(
        dir_path,
        dir_path,
        elec_names=os.path.join(dir_path, 'channel_labels.txt'),
        rereference_grid='array',
        rereference_method='avg',
        bands=['raw'],
        intermediate_fs=100,
        final_fs=100,
        store_reference=True,
        aud_fn=None,
        befaft=[1,1]
    )

    # check label loading
    assert tuple(data_out.info['channel_labels']) == ('Ch1', 'Ch2')

    # check extracted data is correctly rereferenced
    assert np.allclose(data_out['raw'][0][100:600,0], np.zeros((500,)))
    assert np.allclose(data_out[0]['reference'][:,0], true_data['data'][1900:3100].mean(1))
    
    data_out = process_ieeg(
        dir_path,
        dir_path,
        elec_names=['A1', 'B1'],
        rereference_grid='array',
        rereference_method='avg',
        bands=['raw'],
        intermediate_fs=100,
        final_fs=100,
        store_reference=True,
        aud_fn=None,
        befaft=[1,1]
    )

    # check label loading
    assert tuple(data_out.info['channel_labels']) == ('A1', 'B1')

    # check extracted data is correctly rereferenced
    assert np.allclose(data_out['raw'][0][100:600,0], np.zeros((500,)))
    assert np.allclose(data_out[0]['reference'], true_data['data'][1900:3100])

    data_out = process_ieeg(
        dir_path,
        dir_path,
        elec_names=['A1', 'B1'],
        rereference_grid='subject',
        rereference_method='avg',
        bands=['raw'],
        intermediate_fs=100,
        final_fs=100,
        store_reference=True,
        aud_fn=None,
        befaft=[1,1]
    )

    # check extracted data is correctly rereferenced
    assert np.allclose(data_out['raw'][0][100:600,0], np.zeros((500,)))
    assert np.allclose(data_out[0]['reference'][:,0], true_data['data'][1900:3100].mean(1))

    data_out = process_ieeg(
        dir_path,
        dir_path,
        elec_inds=[1],
        elec_names=['B1'],
        rereference_grid='subject',
        rereference_method='avg',
        bands=['raw'],
        intermediate_fs=100,
        final_fs=100,
        store_reference=True,
        aud_fn=None,
        befaft=[1,1]
    )

    # check extracted data is correctly rereferenced
    assert np.allclose(data_out['raw'][0][100:600,0], np.zeros((500,)))
    assert np.allclose(data_out[0]['reference'][:,0], true_data['data'][1900:3100,1])


def test_single_stimuli_pipeline_with_rereference_downsample(small_data_fs50):
    """This data has a sampling rate of 50, which is different from wav file stimulus"""
    dir_path = small_data_fs50['path']

    data_out = process_ieeg(
        dir_path,
        dir_path,
        rereference_grid=np.ones((2,2)),
        rereference_method='pca',
        intermediate_fs = 40,
        final_fs = 10,
        bands=['raw'],
        store_reference=True,
        aud_fn=None,
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


def test_single_stimuli_pipeline_with_custom_spectrogram(small_data_fs50):
    """This data has a sampling rate of 50, which is different from wav file stimulus"""
    dir_path = small_data_fs50['path']

    func = lambda x, sr, **kwargs: scipy.signal.spectrogram(x, sr, **kwargs)[2].T
    func_kwargs = {'nperseg': 256, 'noverlap': 96}
    partial_func = partial(func, **func_kwargs)

    result = np.array([
        0.00000000e+00, 5.52224112e-04, 2.14351760e-03, 4.58641676e-03,
        7.59611558e-03, 1.08276187e-02, 1.39198815e-02, 1.65406112e-02,
        1.84254330e-02, 1.94061846e-02, 1.94249656e-02, 1.85327381e-02,
        1.68737024e-02, 1.46586522e-02, 1.21320086e-02, 9.53782536e-03,
        7.08967121e-03, 4.94856108e-03, 3.21112783e-03, 1.90871384e-03,
        1.01608236e-03, 4.67111968e-04, 1.73986919e-04, 4.62254320e-05,
        6.36694267e-06, 1.18926238e-07, 2.18184741e-08, 2.98976988e-06,
        2.31716349e-05, 8.22788425e-05, 1.99748800e-04, 3.84858198e-04,
        6.32261916e-04, 9.21573082e-04, 1.22069870e-03, 1.49188971e-03,
        1.69899245e-03, 1.81426771e-03, 1.82334194e-03, 1.72739069e-03,
        1.54226436e-03, 1.29491836e-03, 1.01806340e-03, 7.44212186e-04,
        5.00354392e-04, 3.04253714e-04, 1.62944882e-04, 7.35113499e-05,
        2.57345619e-05, 5.85585576e-06, 5.17796025e-07, 3.91541882e-10,
        9.63362154e-08, 2.32793604e-06, 1.26040932e-05, 3.87632754e-05,
        8.76747872e-05, 1.62631361e-04, 2.61660607e-04, 3.77143122e-04,
        4.96806460e-04, 6.05848734e-04, 6.89698849e-04, 7.36789720e-04,
        7.40740797e-04, 7.01492536e-04, 6.25180779e-04, 5.22818824e-04,
        4.08109336e-04, 2.94880272e-04, 1.94698761e-04, 1.15142589e-04,
        5.90502350e-05, 2.48299275e-05, 7.67633901e-06, 1.34762774e-06,
        5.29904582e-08, 2.04437384e-10, 2.56831981e-07, 2.75635102e-06,
        1.14895292e-05, 3.11131662e-05, 6.53403840e-05, 1.15519120e-04,
        1.79753682e-04, 2.52791098e-04, 3.26711131e-04, 3.92274349e-04,
        4.40628210e-04, 4.64996352e-04, 4.61978256e-04, 4.32181347e-04,
        3.80057318e-04, 3.12990713e-04, 2.39852438e-04, 1.69342544e-04,
        1.08478955e-04, 6.15490353e-05, 2.97220959e-05, 1.13683545e-05,
        2.96655298e-06, 3.53844143e-07, 2.38109732e-09, 1.19359393e-08,
        5.85879320e-07, 3.88741182e-06, 1.33245248e-05, 3.24497960e-05,
        6.37527992e-05, 1.07649292e-04, 1.61924516e-04, 2.21782539e-04,
        2.80510925e-04, 3.30628682e-04, 3.65269982e-04, 3.79501464e-04,
        3.71281465e-04, 3.41856037e-04, 2.95498932e-04, 2.38664070e-04,
        1.78736984e-04, 1.22661353e-04, 7.57407543e-05, 4.08680280e-05,
        1.83328120e-05, 6.22197967e-06, 1.29176203e-06, 8.31724165e-08,
        0.00000000e+00,
    ])

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
        aud_fn=func,
        aud_kwargs=func_kwargs,
        befaft=[1,1]
    )

    # check custom spectrogram shape and content
    spec = data_out['aud_copy'][0]
    assert spec.shape == (1200, 129)
    assert np.allclose(spec.max(0), result)

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
        aud_fn=partial_func,
        befaft=[1,1]
    )

    # check custom spectrogram shape and content
    spec = data_out['aud_copy'][0]
    assert spec.shape == (1200, 129)
    assert np.allclose(spec.max(0), result)

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
        aud_fn={'ext': partial_func},
        befaft=[1,1]
    )

    # check custom spectrogram shape and content
    spec = data_out['aud_copy ext'][0]
    assert spec.shape == (1200, 129)
    assert np.allclose(spec.max(0), result)

