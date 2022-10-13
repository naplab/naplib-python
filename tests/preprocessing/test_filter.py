import pytest
import numpy as np
import scipy 


from naplib import Data
from naplib.preprocessing import filter_butter, filter_line_noise

def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(1)
    x = 5+3*rng.random(size=(5000,4))
    x2 = 1+2*rng.random(size=(4000,4))
    x3 = 6+0.1*rng.random(size=(3000,4))
    data_tmp = []
    for i, xx in enumerate([x,x2,x3]):
        data_tmp.append({'name': i, 'sound': 0, 'soundf': 100, 'resp': xx, 'dataf': 100, 'befaft': np.array([1.,1.])})
    return {'out': Data(data_tmp), 'x': [x,x2,x3]}

@pytest.fixture(scope='module')
def data60():
    rng = np.random.default_rng(1)
    line_noise = 60.
    samples = np.linspace(0, 10, 5000)
    samples2 = np.linspace(0.23, 10.23, 5000)
    x = 1+rng.random(size=(5000,4))
    x[:,0] = x[:,0] + np.sin(2 * np.pi * line_noise * samples)
    x[:,1] = x[:,1] + np.sin(2 * np.pi * line_noise * samples2)
    x2 = 1+2*rng.random(size=(4000,4))
    data_tmp = []
    for i, xx in enumerate([x,x2]):
        data_tmp.append({'name': i, 'resp': xx, 'dataf': 500, 'befaft': np.array([1.,1.])})
    
    return {'out': Data(data_tmp), 'x': [x,x2]}

# filter line noise

def test_remove_60hz(data60):
    data_tmp = data60['out']

    data_filt = filter_line_noise(data_tmp)

    assert len(data_filt) == len(data_tmp)

    bp_before = bandpower(data_tmp[0]['resp'][:,0], 500, 59, 61)
    bp_after = bandpower(data_filt[0][:,0], 500, 59, 61)
    assert np.allclose(bp_before, 0.4943959279403939)
    assert np.allclose(bp_after, 0.005800069917108402)

    bp_before = bandpower(data_tmp[0]['resp'][:,1], 500, 59, 61)
    bp_after = bandpower(data_filt[0][:,1], 500, 59, 61)
    assert np.allclose(bp_before, 0.4996227062895015)
    assert np.allclose(bp_after, 0.0057041888335567295)

def test_remove_60hz_multiple_num_repeats(data60):
    data_tmp = data60['out']

    data_filt = filter_line_noise(data_tmp, f=60., num_repeats=3)

    bp_before = bandpower(data_tmp[0]['resp'][:,0], 500, 59, 61)
    bp_after = bandpower(data_filt[0][:,0], 500, 59, 61)
    assert np.allclose(bp_before, 0.4943959279403939)
    assert np.allclose(bp_after, 8.432305430784527e-05)

    bp_before = bandpower(data_tmp[0]['resp'][:,1], 500, 59, 61)
    bp_after = bandpower(data_filt[0][:,1], 500, 59, 61)
    assert np.allclose(bp_before, 0.4996227062895015)
    assert np.allclose(bp_after, 7.784150031477097e-05)

def test_filter_line_noise_bad_line_noise(data60):
    data_tmp = data60['out']

    with pytest.raises(AssertionError):
        data_filt = filter_line_noise(data_tmp, f=[[60]])

# butterworth filter

def test_lowpass_Data(data):
    filtered = filter_butter(data['out'], field='resp', Wn=10, btype='lowpass')
    bp_low = bandpower(filtered[0][:,0], 100, 0.1, 10)
    bp_high = bandpower(filtered[0][:,0], 100, 10, 19.9)
    assert np.allclose(bp_low, 0.11533344625202704)
    assert np.allclose(bp_high, 0.0070457852200255045)

def test_highpass(data):
    filtered = filter_butter(field=data['out']['resp'], fs=data['out']['dataf'][0], Wn=10, btype='highpass')
    bp_low = bandpower(filtered[0][:,0], 100, 0.1, 10)
    bp_high = bandpower(filtered[0][:,0], 100, 10, 19.9)
    print((bp_low, bp_high))
    assert np.allclose(bp_low, 0.004879470143977822)
    assert np.allclose(bp_high, 0.09462148354354974)

def test_bandpass(data):
    filtered = filter_butter(field=data['out']['resp'], fs=data['out']['dataf'][0], Wn=[10, 20], btype='bandpass')
    bp_low = bandpower(filtered[0][:,0], 100, 0.1, 10)
    bp_band = bandpower(filtered[0][:,0], 100, 10, 20)
    bp_high = bandpower(filtered[0][:,0], 100, 20, 29.9)
    assert np.allclose(bp_low, 0.0021222407044427945)
    assert np.allclose(bp_band, 0.10979631639822579)
    assert np.allclose(bp_high, 0.0053699618743934365)

def test_bandstop(data):
    filtered = filter_butter(field=data['out']['resp'], fs=data['out']['dataf'][0], Wn=[10, 20], btype='bandstop')
    bp_low = bandpower(filtered[0][:,0], 100, 0.1, 10)
    bp_band = bandpower(filtered[0][:,0], 100, 10, 20)
    bp_high = bandpower(filtered[0][:,0], 100, 20, 29.9)
    assert np.allclose(bp_low, 0.1294933044166706)
    assert np.allclose(bp_band, 0.005552836561214947)
    assert np.allclose(bp_high, 0.11518997948303428)

def test_return_filters(data):
    b_true = np.array([ 0.06745527,  0.        , -0.13491055,  0.        ,  0.06745527])
    a_true = np.array([ 1.        , -1.94246878,  2.1192024 , -1.21665164,  0.4128016 ])
    filtered, filters = filter_butter(field=data['out']['resp'], fs=data['out']['dataf'][0], Wn=[10, 20], btype='bandpass', return_filters=True)
    assert isinstance(filters, list)
    assert len(filters) == 3
    assert np.allclose(filters[0][0], b_true)
    assert np.allclose(filters[0][1], a_true)
