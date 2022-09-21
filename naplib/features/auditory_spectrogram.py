from os.path import dirname, join
import math
import numpy as np
from scipy.signal import resample, lfilter
from hdf5storage import loadmat

def _largest_pow2_less_than(x):
    '''
    Get largest power of 2 less than x.
    '''
    return 2**(math.ceil(math.log(x, 2)) - 1)

def _sigmoid(x, factor):
    '''
    Nonlinear sigmoid for cochlear model which simulates hair
    cell nonlinearity.
    
    Parameters
    ----------
    x : float
        Data to pass through nonlinear function
    factor: float
        Nonlinear factor.
        If >0, transistor-like function. 
        If =0, hard-limiter
        If =-1, half-wave rectifier
        Else, no operation and x is returned unchanged, i.e. linear
    '''
    if isinstance(factor, float):
        return 1 / (1 + np.exp(-x / factor))
    if factor == 'boolean':
        return (x>0).astype('float')
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
        Sampling rate of the signal. For best performance, it may be
        recommended to resample the signal to a power of 2 sampling rate.
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
        
    if (isinstance(factor, float) and factor <= 0) or (isinstance(factor, int) and factor <= 0):
        raise ValueError(f'If a float, factor must be positive, but got {factor}')
    if isinstance(factor, str) and factor not in ['linear', 'boolean', 'half-wave']:
        raise ValueError(f"If a string, factor must be one of {'linear', 'boolean', 'half-wave'}, but got '{factor}'")
        
    # ispow2 = (sfreq and (not(sfreq & (sfreq - 1))) )
        
    # if not ispow2:
    #     next_pow2 = _largest_pow2_less_than(sfreq)
    #     x_resampled = resample(x.squeeze(), int(x.shape[0]/sfreq*next_pow2))
    # else:
    x_resampled = x.squeeze().copy()

        
    shift = round(math.log2(sfreq/16384.)) # octaves to shift
    
    # read in cochlear filter from file
    filedir = dirname(__file__)
    # filedir = './'
    cochba = loadmat(join(filedir, 'cochba.mat'), variable_names=['COCHBA'])
    cochba = cochba['COCHBA']
    
    L, M = cochba.shape
    L_x = x_resampled.shape[0]
    
    L_frm = round(frame_len * (2**(4+shift))) # frame length (points)


    
    if tc > 0:
        alpha = math.exp(-1/(tc * 2**(4+shift))) # decay factor
        alpha_filt = np.array([1, -alpha])
    else:
        alpha = 0 # this is now just short-term average
        
    # hair cell time constant in ms
    haircell_tc = 0.5
    beta = math.exp(-1/(haircell_tc * 2**(4+shift)))
    low_pass_filt = np.array([1, -beta])
    
    # allocate memory for output
    N = math.ceil(L_x / L_frm)
    x_resampled = np.pad(x_resampled, (0,N*L_frm-L_x))
    indexer = np.arange(L_frm-1, N*L_frm, L_frm)
    
    output = np.zeros((N, M-1))
    
    # do last channel first (highest frequency)
    
    p = int(np.real(cochba[0,M-1]))
    B = np.real(cochba[1:p+2, M-1])
    A = np.imag(cochba[1:p+2, M-1])
    
    y1 = lfilter(B, A, x_resampled).squeeze()
    y2 = _sigmoid(y1, factor)
        
    # hair cell membrane (low-pass <= 4kHz), ignored for linear ionic channels
    if factor != 'linear':
        y2 = lfilter(np.array([1.]), low_pass_filt, y2).squeeze()
    
    y2_h = y2.copy()
    y3_h = 0
    
    for ch in range(M-2, -1, -1):
        
        # cochlear filterbank
        p = int(np.real(cochba[0,ch]))
        B = np.real(cochba[1:p+2, ch])
        A = np.imag(cochba[1:p+2, ch])
        y1 = lfilter(B, A, x_resampled).squeeze()
    
        # transduction hair cells
        y2 = _sigmoid(y1, factor)
        
        # hair cell membrane (low-pass <= 4kHz), ignored for linear ionic channels
        if factor != 'linear':
            y2 = lfilter(np.array([1.]), low_pass_filt, y2).squeeze()
    
        # reduction: lateral inhibitory network
        # masked by higher (frequency) spatial response
        y3 = y2 - y2_h
        y2_h = y2.copy()
        
        # half-wave rectifier
        y4 = np.maximum(y3, 0)
        
        # temporal integration
        if alpha != 0:
            y5 = lfilter(np.array([1.]), alpha_filt, y4).squeeze()
            output[:, ch] = y5[indexer]
        else:
            if L_frm == 1:
                output[:, ch] = y4
            else:
                output[:, ch] = np.mean(np.reshape(y4, (L_frm, N)), axis=0)
        
    return output
    
    