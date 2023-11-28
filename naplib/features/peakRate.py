import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import zscore


def peakRate(aud, sfreq, band=[1,10], th=0):
    '''
    Extract peakRate events from the (auditory) spectrogram
    
    Parameters
    ----------
    aud : np.ndarray
        Auditory spectrogram, shape (n_frames, n_frequency)
    sfreq : int
        Sampling rate of the spectrogram
    band : List, default=[1,10]
        Envelope bandpass filter bandwidth
    th: float, default=0
        Threshold (in units of standard deviations) of peaks to keep

    Returns
    -------
    peakRate : np.ndarray
        Time-series of peakRate events, shape (n_frames, )

    See Oganian, Y., & Chang, E. F. (2019). A speech envelope landmark for
    syllable encoding in human superior temporal gyrus. Science advances
    '''
    
    # Extract the envelope by taking the mean over frequencies
    env = np.mean(aud, axis=1)

    # Bandpass to 1-10 Hz
    b, a = butter(3, band, btype='bandpass', fs=sfreq)
    env = zscore(filtfilt(b, a, env))

    # Temporal derivative and ReLU
    rate = np.maximum(np.diff(env, prepend=env[0]), 0)

    # Extract local peaks
    peakRate = np.zeros_like(env)
    peaks = find_peaks(rate, height=th*np.std(rate))[0]
    peakRate[peaks] = rate[peaks]

    return peakRate