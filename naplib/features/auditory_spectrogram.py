from os.path import dirname, join
import math
import numpy as np
from scipy.signal import resample, lfilter
from hdf5storage import loadmat

from naplib import logger


# read in cochlear filter from file
COCHBA = loadmat(
    join(dirname(__file__), 'cochba.mat'), variable_names=['COCHBA']
)['COCHBA']


def _sigmoid(x, factor):
    '''
    Nonlinear sigmoid for cochlear model which simulates hair
    cell nonlinearity.
    
    Parameters
    ----------
    x : float
        Data to pass through nonlinear function
    factor: float or string
        Nonlinear factor.
        If a number > 0, transistor-like function. 
        If 'boolean', hard-limiter
        If 'half-wave', half-wave rectifier
        Else, no operation and x is returned unchanged, i.e. linear
    '''
    if isinstance(factor, float):
        return 1 / (1 + np.exp(-x / factor))
    if factor == 'boolean':
        return (x > 0).astype('float')
    elif factor == 'half-wave':
        return np.maximum(x, 0)
    else:
        return x


def auditory_spectrogram(x, sfreq, frame_len=8, tc=4, factor='linear'):
    '''
    Compute the auditory spectrogram of a signal using a model of the
    peripheral auditory system [#1yang]_. The model includes a cochlear
    filter bank of 128 constant-Q filters logarithmically-spaced, a hair
    cell model which includes a low-pass filter and a nonlinear compression
    function, and a lateral inhibitory network over the spectral axis. The
    envelope of each frequency band gives the time-frequency representation.
    
    Parameters
    ----------
    x : np.ndarray
        Acoustic signal to convert to time-frequency representation
    sfreq : int
        Sampling rate of the signal. This function is meant to be used on a signal
        with 16KHz sampling rate. If sfreq is different, it resamples the audio to 16KHz.
    frame_len : float, default=8
        Frame length of the output, in ms. Typically 8, 16, or a power of 2.
    tc : float, default=4
        Time constant for leaky integration. Must be >=0. If 0, then leaky
        integration becomes short-term average. Typically 4, 16, or 64 ms.
    factor : float or string, default='linear'
        Nonlinear factor for hair cell model. If a positive float, specifies the
        critical level factor (typically 0.1 for a unit sequence). The smaller the
        value, the more the compression. Or, can be one of
        {'linear', 'boolean', 'half-wave'} describing the compressor.
    
    Returns
    -------
    aud : np.ndarray
        Auditory spectrogram, shape (n_frames, 128).
    
    Notes
    -----
    This is a python re-implementation of the Matlab function `wav2aud` in the
    `NSLtools toolbox <https://github.com/tel/NSLtools>`_.

    For correct performance, x should be a float array.
    
    References
    ----------
    .. [#1yang] X. Yang, K. Wang and S. A. Shamma, "Auditory representations of
        acoustic signals," in IEEE Transactions on Information Theory, vol.
        38, no. 2, pp. 824-839, March 1992, doi: 10.1109/18.119739.
    
    '''
    
    if frame_len <= 0:
        raise ValueError(f'frame_len must be positive but got {frame_len}')
    
    if tc < 0:
        raise ValueError(f'time constant (tc) must be nonnegative but got {tc}')
    
    if isinstance(factor, (float, int)) and factor <= 0:
        raise ValueError(f'If a float, factor must be positive, but got {factor}')
    if isinstance(factor, str) and factor not in ['linear', 'boolean', 'half-wave']:
        raise ValueError(f"If a string, factor must be one of {'linear', 'boolean', 'half-wave'}, but got '{factor}'")
    
    x = x.squeeze()

    # If x is not sampled at 16KHz, resample
    if sfreq != 16_000:
        logger.warning(f"Resampling audio from {round(sfreq)/1000:g}KHz to 16KHz")
        x, sfreq = resample(x, round(len(x) / sfreq * 16_000)), 16_000

    L_x = x.shape[0]
    L_frm = round(frame_len * 2**4) # frame length (points)
    
    if tc > 0:
        alpha = math.exp(-1 / (tc * 2**4)) # decay factor
        alpha_filt = np.array([1, -alpha])
    else:
        alpha = 0 # this is now just short-term average
    
    # hair cell time constant in ms
    haircell_tc = 0.5
    beta = math.exp(-1 / (haircell_tc * 2**4))
    low_pass_filt = np.array([1, -beta])
    
    # allocate memory for output
    N = math.ceil(L_x / L_frm)
    M = COCHBA.shape[1]
    x = np.pad(x, (0, N*L_frm-L_x))
    indexer = np.arange(L_frm-1, N*L_frm, L_frm)
    output = np.zeros((N, M-1), dtype='float32')
    
    # do last channel first (highest frequency)
    
    p = round(np.real(COCHBA[0, M-1]))
    B = np.real(COCHBA[1:p+2, M-1])
    A = np.imag(COCHBA[1:p+2, M-1])
    
    y = lfilter(B, A, x).squeeze()
    y = _sigmoid(y, factor)
    
    # hair cell membrane (low-pass <= 4kHz), ignored for linear ionic channels
    if factor != 'linear':
        y = lfilter(np.array([1.]), low_pass_filt, y).squeeze()
    
    y_h = y
    
    for ch in range(M-2, -1, -1):
        
        # cochlear filterbank
        p = round(np.real(COCHBA[0, ch]))
        B = np.real(COCHBA[1:p+2, ch])
        A = np.imag(COCHBA[1:p+2, ch])
        y = lfilter(B, A, x).squeeze()
        
        # transduction hair cells
        y = _sigmoid(y, factor)
        
        # hair cell membrane (low-pass <= 4kHz), ignored for linear ionic channels
        if factor != 'linear':
            y = lfilter(np.array([1.]), low_pass_filt, y).squeeze()
        
        # reduction: lateral inhibitory network
        # masked by higher (frequency) spatial response
        y, y_h = y - y_h, y
        
        # half-wave rectifier
        y = np.maximum(y, 0)
        
        # temporal integration
        if alpha != 0:
            y = lfilter(np.array([1.]), alpha_filt, y).squeeze()
            output[:, ch] = y[indexer]
        else:
            if L_frm == 1:
                output[:, ch] = y
            else:
                output[:, ch] = np.mean(np.reshape(y, (L_frm, N)), axis=0)
    
    return output

