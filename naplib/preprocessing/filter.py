import numpy as np
from scipy import signal as sig

from ..data import Data
from ..utils import _parse_outstruct_args


def filter_line_noise(data=None, field='resp', fs='dataf', f=60, num_taps=501, axis=0, num_repeats=1):
    '''
    Filter input data with a notch filter to remove line noise and its harmonics.
    A notch FIR filter is applied at the line noise frequency and all of its
    harmonics up to the Nyquist rate.
    
    Parameters
    ----------
    data : naplib.Data instance, optional
        Data object containing data to be filtered in one of the fields. If None, then must give
        the signals to be filtered in the 'field' argument.
    field : string | list of np.ndarrays or a multidimensional np.ndarray
        Field of trials to filter. If a string, it must specify one of the fields of the Data
        provided in the first argument. If a multidimensional array, first dimension
        indicates the trial/instances which will be concatenated over to compute
        normalization statistics. Each trial is filtered independently.
    fs : string | int, default='dataf'
        Sampling rate of the data. Either a string specifying a field of the Data or an int
        giving the sampling rate for all trials.
    f : float, default=60
        Line noise frequency, in Hz.
    num_taps : int, default=501
        Number of taps in the FIR filter. Must be odd.
    axis : int or None, default=0
        Axis of the array to apply the filter to.
    num_repeats : int, default=1
        Number of times to repeat convolving the filter. This is useful to increase if the number of taps
        is low compared to the sampling rate (e.g. less than 1 full second). 

    Returns
    -------
    filtered_data : list of np.ndarrays

    '''
    
    field, fs = _parse_outstruct_args(data, field, fs, allow_different_lengths=True, allow_strings_without_outstruct=False)
    
    assert isinstance(f, int) or isinstance(f, float), 'line-noise frequency "f" must be an int or float'
    
        
    output = []
    
    # params for firwin2
    fs_ = float(fs[0])
    
    for x, trial_fs in zip(field, fs):
        
        multiplier = 1

        filtered_x = x.copy()
        if filtered_x.ndim == 1:
            filtered_x = filtered_x[:,np.newaxis]

        while multiplier * f < float(trial_fs) / 2:

            notchFreq = multiplier * f
            
            freqs = np.array([0, notchFreq-1, notchFreq-.5, notchFreq+.5, notchFreq+1, fs_/2])/(fs_/2)
            gains = np.array([1, 1, 0, 0, 1, 1])
            taps = sig.firwin2(num_taps, freqs, gains)
            taps_conv = sig.convolve(taps, taps[::-1]).reshape(-1,1)

            for _ in range(num_repeats):
                filtered_x = sig.convolve(filtered_x, taps_conv, mode='same')

            multiplier += 1
    
        output.append(filtered_x)
    
    return output


def filter_butter(data=None, field='resp', btype='bandpass', Wn=[70,150], fs='dataf', order=2, return_filters=False):
    '''
    Filter time series signals using an Nth order digital Butterworth filter. The filter
    is applied to each column of each trial in the field data.
    
    Parameters
    ----------
    data : naplib.Data instance, optional
        Data object containing data to be normalized in one of the field. If not given, must give
        the data to be normalized directly as the ``data`` argument. 
    field : string | list of np.ndarrays or a multidimensional np.ndarray
        Field to bandpass filter. If a string, it must specify one of the fields of the Data
        provided in the first argument. If a multidimensional array, first dimension
        indicates the trial/instances. Each trial's data must be of shape (time, channels)
    Wn : float, list or array-like, default=[70,150]
        Critical frequencies, in Hz. The critical frequency or frequencies. For lowpass and highpass filters,
        Wn is a scalar; for bandpass and bandstop filters, Wn is a length-2 sequence.
    btype : string, defualt='bandpass
        Filter type, one of {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    fs : string | float
        Sampling rate of the field to filter. If a string, must specify a field of the Data
        object. Can be a single float if all trial's have the same sampling rate, or can be
        a list of floats specifying the sampling rate for each trial.
    order : int
        The order of the filter.
    return_filters : bool, default=False
        If True, return the filter transfer function coefficients from each trial's filtering.
    
    Returns
    -------
    filtered_data : list of np.ndarrays
        Filtered time series.
    filter : list
        Filter transfer function coefficients returned as a list of (b, a) tuples. Only
        returned if ``return_filters`` is True.
    '''
    
    field, fs = _parse_outstruct_args(data, field, fs, allow_different_lengths=True, allow_strings_without_outstruct=False)

    if not isinstance(fs, list):
        fs = [fs for _ in field]
        
    filtered_data = []
    
    filters = []
    
    for trial_data, trial_fs in zip(field, fs):
    
        b, a = sig.butter(order, Wn, btype=btype, fs=trial_fs, output='ba')
        
        filters.append((b, a))
        
        filtered_data.append(sig.filtfilt(b, a, trial_data, axis=0))
        
    if return_filters:
        return filtered_data, filters

    return filtered_data
