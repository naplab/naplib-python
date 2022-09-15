import pytest
import numpy as np
import scipy 


from naplib import Data
from naplib.preprocessing import filter_butter

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
