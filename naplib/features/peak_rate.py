import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import zscore


def peak_rate(aud, sfreq, band=[1,10], thresh=0):
    '''
    Extract peak_rate events [#1oganian]_ from the (auditory) spectrogram
    
    Parameters
    ----------
    aud : np.ndarray
        Auditory spectrogram, shape (n_frames, n_frequency)
    sfreq : int
        Sampling rate of the spectrogram
    band : List, default=[1,10]
        Envelope bandpass filter bandwidth
    thresh: float, default=0
        Threshold (in units of standard deviations) of peaks to retain

    Returns
    -------
    peak_rate : np.ndarray
        Time-series of peak_rate events, shape (n_frames, )

    References
    ----------
    .. [#1oganian] Y. Oganian, & E.F. Chang, "A speech envelope landmark for
        syllable encoding in human superior temporal gyrus," in Science Advances,
        vol 5, no. 11, pp. eaay6279, November 2019, doi: 10.1126/sciadv.aay6279
    '''
    
    # Extract the envelope by taking the mean over frequencies
    env = np.mean(aud, axis=1)

    # Bandpass to 1-10 Hz
    b, a = butter(3, band, btype='bandpass', fs=sfreq)
    env = zscore(filtfilt(b, a, env))

    # Temporal derivative and ReLU
    rate = np.maximum(np.diff(env, prepend=env[0]), 0)

    # Extract local peaks
    peak_rate = np.zeros_like(env)
    peaks = find_peaks(rate, height=thresh*np.std(rate))[0]
    peak_rate[peaks] = rate[peaks]

    return peak_rate