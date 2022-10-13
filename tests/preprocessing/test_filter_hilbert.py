import pytest
import numpy as np
import scipy 


from naplib import Data
from naplib.preprocessing import filterbank_hilbert, phase_amplitude_extract

@pytest.fixture(scope='module')
def data_hilbert():
    rng = np.random.default_rng(1)
    samples = np.linspace(0, 10, 5000)
    samples2 = np.linspace(0.23, 10.23, 5000)
    x = 1+rng.random(size=(5000,4))
    x = x + (np.sin(2 * np.pi * 20 * samples) + np.sin(2 * np.pi * 30 * samples) + np.sin(2 * np.pi * 75 * samples) + np.sin(2 * np.pi * 90 * samples)).reshape(-1,1)
    x2 = 1+2*rng.random(size=(4000,4))
    data_tmp = []
    for i, xx in enumerate([x,x2]):
        data_tmp.append({'name': i, 'resp': xx, 'dataf': 500, 'befaft': np.array([1.,1.])})
        
    return {'out': Data(data_tmp), 'x': [x,x2]}


# phase_amplitude_extract tests

def test_set_bandnames(data_hilbert):
    data_tmp = data_hilbert['out']

    phs_amp_data = phase_amplitude_extract(data_tmp, 'resp', Wn=np.array([[8,13],[15,30],[70,100]]), bandnames=['bandA','bandB','bandC'])
    assert phs_amp_data.fields == ['bandA phase', 'bandA amp', 'bandB phase', 'bandB amp', 'bandC phase', 'bandC amp']

def test_set_bandnames_duplicates(data_hilbert):
    data_tmp = data_hilbert['out']

    with pytest.raises(ValueError):
        phs_amp_data = phase_amplitude_extract(data_tmp, 'resp', Wn=[[8,13],[15,30],[70,100]], bandnames=['bandA','bandB','bandB'])

def test_set_bands_duplicates(data_hilbert):
    data_tmp = data_hilbert['out']

    with pytest.raises(ValueError):
        phs_amp_data = phase_amplitude_extract(data_tmp, 'resp', Wn=[[8,13],[8,13],[70,100]], bandnames=['bandA','bandB','bandC'])

def test_frequency_range_too_narrow_extract(data_hilbert):
    data_tmp = data_hilbert['out']

    with pytest.raises(ValueError) as excinfo:
        phs_amp_data = phase_amplitude_extract(data_tmp, 'resp', Wn=[[8,13],[40,42],[70,100]], bandnames=['bandA','bandB','bandC'])
    assert 'is too narrow and no filters ' in str(excinfo.value)

def test_filter_multiple_bands_same_as_one(data_hilbert):
    data_tmp = data_hilbert['out']

    phs_amp_data = phase_amplitude_extract(data_tmp, 'resp', Wn=np.array([[8,13],[15,30],[70,100]]))
    assert len(phs_amp_data) == len(data_tmp)

    phs_amp_data2 = phase_amplitude_extract(data_tmp, 'resp', Wn=[70,100])
    assert len(phs_amp_data2) == len(phs_amp_data)

    same_fields = phs_amp_data2.fields

    for field in same_fields:
        for trial1, trial2 in zip(phs_amp_data[field], phs_amp_data2[field]):
            assert np.allclose(trial1, trial2)

    assert np.allclose(phs_amp_data['[ 8 13] amp'][0][1000,:], np.array([0.02395993, 0.05828907, 0.07257716, 0.02046574]))
    assert np.allclose(phs_amp_data['[ 8 13] phase'][0][1000,:], np.array([ 1.14940659, -1.80128736, -2.42091546,  1.40372582]))
    assert np.allclose(phs_amp_data['[15 30] amp'][0][1000,:], np.array([0.69737044, 0.61818003, 0.67543224, 0.66870002]))
    assert np.allclose(phs_amp_data['[15 30] phase'][0][1000,:], np.array([-1.48205289, -1.4974508 , -1.57238392, -1.40474137]))
    assert np.allclose(phs_amp_data['[ 70 100] amp'][0][1000,:], np.array([0.76066884, 0.74351326, 0.77783409, 0.73482941]))
    assert np.allclose(phs_amp_data['[ 70 100] phase'][0][1000,:], np.array([-1.31277148, -1.37329027, -1.34475956, -1.36207981]))


def test_bad_signal_shape():
    x = np.random.rand(1000, 5, 3)
    fs=100

    with pytest.raises(ValueError) as excinfo:
        x_phs, x_amp, cfs = filterbank_hilbert(x, fs, Wn=[1, 150])
    assert 'Input signal must be 1- or 2-dimensional' in str(excinfo.value)

# tests filterbank_hilbert

def test_freq_range_inverted():
    x = np.random.rand(1000, 5)
    fs = 100

    with pytest.raises(ValueError) as excinfo:
        phs_amp_data = filterbank_hilbert(x, fs, Wn=[13, 8])
    assert 'must be greater than lower bound' in str(excinfo.value)

def test_no_filters_in_freq_range():
    x = np.random.rand(10000, 5)
    fs = 500

    with pytest.raises(ValueError) as excinfo:
        phs_amp_data = filterbank_hilbert(x, fs, Wn=[40, 42])
    assert 'is too narrow, so' in str(excinfo.value)

def test_center_freqs_and_output_shape():
    x = np.random.rand(1000, 5)
    fs = 100
    expected_cfs = np.array([  1.21558792,   1.64557736,   2.1458696 ,   2.71717229,
          3.36004234,   4.07492865,   4.499086  ,   4.96739367,
          5.48444726,   6.05532071,   6.6856161 ,   7.38151862,
          8.14985731,   8.99817199,   9.93478734,  10.96889452,
         12.11064142,  13.37123219,  14.76303725,  16.29971462,
         17.99634398,  19.86957468,  21.93778904,  24.22128283,
         26.74246438,  29.5260745 ,  32.59942923,  35.99268797,
         39.73914935,  43.87557808,  48.44256567,  53.48492877,
         59.05214899,  65.19885846,  71.98537593,  79.4782987 ,
         87.75115616,  96.88513133, 106.96985753, 118.10429798,
        130.39771692, 143.97075186])

    x_phs, x_amp, cfs = filterbank_hilbert(x, fs, Wn=[1, 150])

    assert x_phs.shape == (*x.shape, len(expected_cfs))
    assert x_amp.shape == (*x.shape, len(expected_cfs))
    assert np.allclose(cfs, expected_cfs)

def test_oneD_signal_same_output():
    x = np.random.rand(1000, 1)
    fs = 100

    x_phs, x_amp, cfs = filterbank_hilbert(x, fs, Wn=[10, 20])
    x_phs2, x_amp2, cfs2 = filterbank_hilbert(x.squeeze(), fs, Wn=[10, 20])

    assert np.allclose(x_phs, x_phs2)
    assert np.allclose(x_amp, x_amp2)
    assert np.allclose(cfs, cfs2)

def test_bad_x_shape():
    x = np.random.rand(1000, 5, 3)
    fs=100

    with pytest.raises(ValueError):
        x_phs, x_amp, cfs = filterbank_hilbert(x, fs, Wn=[1, 150])


