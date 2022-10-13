import numpy as np
from joblib import Parallel, delayed
from scipy.fft import fft, ifft

from ..data import Data
from ..utils import _parse_outstruct_args


def phase_amplitude_extract(data=None, field='resp', fs='dataf', Wn=[[70, 150]], bandnames=None, n_jobs=-1):
    '''
    Extract phase and amplitude (envelope) from a frequency band or a set of frequency bands.
    This is done by averaging over the envelopes and phases computed from the Hilbert Transform
    of each filter output in a filterbank of bandpass filters [#1edwards]_. 
    
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
    Wn : list or array-like, shape (n_freq_bands, 2) or (2,), default=[[70, 150]]
        Lower and upper boundaries for filterbank center frequencies. The default
        of [[70, 150]] extracts the phase and amplitude of the highgamma band only.
    bandnames : list of strings, length=n_freq_bands, optional
        If provided, these are used to create the field names for each frequency band's amplitude
        and phase in the output Data. Should be the same length as the number of bands
        specified in Wn. For example, if Wn=[[8,12],[70,150]], and bandnames=['theta','highgamma'],
        then the fields of the output Data will be {'theta phase', 'theta amp', 'highgamma phase',
        'highgamma amp'}. But if bandnames=None, then they will be {'[ 8 12] phase', '[ 8 12] amp',
        '[ 70 150] phase', '[ 70 150] amp'}.
    n_jobs : int, default=-1
        Number of jobs to use to compute filterbank across channels in parallel in filterbank_hilbert.

    Returns
    -------
    phase_amplitude_data : naplib.Data instance
        An instance of naplib.Data containing 2*n_freq_bands fields. For each frequency band,
        there will be a field for phase and a field for amplitude of that band, each with
        shape (time, channels) for each trial.
    

    See Also
    --------
    filterbank_hilbert

    References
    ----------
    .. [#1edwards] Edwards, Erik, et al. "Comparison of time–frequency responses
               and the event-related potential to auditory speech stimuli in
               human cortex." Journal of neurophysiology 102.1 (2009): 377-386.

    '''
    field, fs = _parse_outstruct_args(data, field, fs, allow_different_lengths=True, allow_strings_without_outstruct=False)
    
    Wn_ = np.asarray(Wn)
    if Wn_.ndim == 1:
        Wn_ = Wn_[np.newaxis,:]
    
    max_f_overall = Wn_.max() + 1
    
    
    n_freq_bands = Wn_.shape[0]
    freq_band_names = []
    if bandnames is not None:
        if not (isinstance(bandnames, list) and isinstance(bandnames[0], str) and len(bandnames) == Wn_.shape[0]):
            raise ValueError(f'If provided, bandnames must be a list of strings, one for each frequency band.')
        if len(set(bandnames)) != len(bandnames):
            raise ValueError(f'All frequency bandnames must be unique, but found duplicates.')
        for bandname in bandnames:
            freq_band_names.append(bandname + ' phase')
            freq_band_names.append(bandname + ' amp')
    else:
        for freq_band in Wn_:
            freq_band_names.append(f'{freq_band} phase')
            freq_band_names.append(f'{freq_band} amp')

    if len(set([str(w) for w in Wn_])) != Wn_.shape[0]:
        raise ValueError(f'All frequency bands in Wn must be unique, but found duplicates.')

    # prepare dict of lists to put output data into
    phase_amplitude_data = {}
    for freq_band_name in freq_band_names:
        phase_amplitude_data[freq_band_name] = []

    # loop through trials
    for trial, fs_trial in zip(field, fs):
        x_phase, x_amplitude, center_freqs = filterbank_hilbert(trial, fs_trial, Wn=[0, max_f_overall], n_jobs=n_jobs)
        
        for ii, freq_band in enumerate(Wn_):
            minf, maxf = freq_band
            band_locator = np.logical_and(center_freqs>=minf, center_freqs<=maxf)
            if band_locator.sum() == 0:
                raise ValueError(f'Frequency band {freq_band} is too narrow and no filters have center frequencies inside it. Try a wider frequency band.')
            # average over the filters in the frequency band
            phase_amplitude_data[freq_band_names[2*ii]].append(x_phase[:,:,band_locator].mean(-1))
            phase_amplitude_data[freq_band_names[2*ii+1]].append(x_amplitude[:,:,band_locator].mean(-1))
    
    return Data(phase_amplitude_data, strict=False)
    
    
def filterbank_hilbert(x, fs, Wn=[1,150], n_jobs=-1):
    '''
    Compute the phase and amplitude (envelope) of a signal over a range of frequencies,
    as in [#1edwards]_. This is done using a filter bank of gaussian shaped filters with
    center frequencies linearly spaced until 4Hz and then logarithmically spaced. The
    Hilbert Transform of each filter's output is computed and the real and imaginary parts
    form the amplitude and phase, respectively. See [#1edwards]_ for details on the filter
    bank used.
    
    Parameters
    ----------
    x : np.ndarray, shape (time, channels)
        Signal to filter. Filtering is performed on each channel independently.
    fs : int
        Sampling rate.
    Wn : list or array-like, length 2, default=[1, 150]
        Lower and upper boundaries for filterbank center frequencies. The default
        of [1, 150] results in 42 filters.
    n_jobs : int, default=-1
        Number of jobs to use to compute filterbank across channels in parallel.
    
    Returns
    -------
    x_phase : np.ndarray, shape (time, channels, frequency_bins)
        Phase of each frequency bin in the filter bank for each channel.
    x_envelope : np.ndarray, shape (time, channels, frequency_bins)
        Envelope of each frequency bin in the filter bank for each channel.
    center_freqs : np.ndarray, shape (frequency_bins,)
        Center frequencies for each frequency bin used in the filter bank.

    Examples
    --------
    >>> import naplib as nl
    >>> from naplib.preprocessing import filterbank_hilbert as fb_hilb
    >>> import numpy as np
    >>> x = np.random.rand(1000,3) # 3 channels of signals
    >>> fs = 500
    >>> x_phase, x_envelope, freqs = fb_hilb(x, fs, Wn=[1, 150])
    >>> # the outputs have the phase and envelope for each channel and each filter in the filterbank
    >>> x_phase.shape  # 3rd dimension is one for each filter in filterbank
    (1000, 3, 42)
    >>> x_envelope.shape
    (1000, 3, 42)
    >>> freqs[0] # center frequency of first filter bank filter
    1.21558792
    >>> freqs[-1] # center frequency of last filter bank filter
    143.97075186

    
    References
    ----------
    .. [#1edwards] Edwards, Erik, et al. "Comparison of time–frequency responses
               and the event-related potential to auditory speech stimuli in
               human cortex." Journal of neurophysiology 102.1 (2009): 377-386.
    
    '''
    
    # create filter bank
    a = np.array([np.log10(0.39), 0.5])
    f0          = 0.018 
    octSpace    = 1./7 
    minf, maxf  = Wn
    if minf >= maxf:
        raise ValueError(f'Upper bound of frequency range must be greater than lower bound, but got lower bound of {minf} and upper bound of {maxf}')
    maxfo       = np.log2(maxf/f0)  # octave of max freq
    
    cfs         = [f0]
    sigma_f     = 10**(a[0]+a[1]*np.log10(cfs[-1]))
    
    if x.ndim != 1 and x.ndim != 2:
        raise ValueError(f'Input signal must be 1- or 2-dimensional but got input with shape {x.shape}')
    
    if x.ndim == 1:
        x = x[:,np.newaxis]
        
    while np.log2(cfs[-1]/f0) < maxfo:
        
        if cfs[-1] < 4:
            cfs.append(cfs[-1]+sigma_f)
        else: # switches to log spacing at 4 Hz
            cfo = np.log2(cfs[-1]/f0)        # current freq octave
            cfo += octSpace           # new freq octave
            cfs.append(f0*(2**(cfo)))
        
        sigma_f = 10**(a[0]+a[1]*np.log10(cfs[-1]))
        
    cfs = np.array(cfs)
    if np.logical_and(cfs>=minf, cfs<=maxf).sum() == 0:
        raise ValueError(f'Frequency band is too narrow, so no filters in filterbank are placed inside. Try a wider frequency band.')
    
    cfs = cfs[np.logical_and(cfs>=minf, cfs<=maxf)] # choose those that lie in the input freqRange

    
    exponent = np.concatenate((np.ones((len(cfs),1)), np.log10(cfs)[:,np.newaxis]), axis=1) @ a
    sigma_fs = 10**exponent
    sds = sigma_fs * np.sqrt(2)
    
    N = x.shape[0]
    freqs  = np.arange(0, N//2+1)*(fs/N)
    
    # perform hilbert transform at each center freq
    
    Xf = fft(x, N, axis=0)
    
    h = np.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[0] = slice(None)
        h = h[tuple(ind)]   

    filtered_data = np.zeros((*x.shape, len(cfs)), dtype=np.cdouble)


    def _vectorized_band_hilbert(X_fft, h_, N_, freqs_, cfs_, sds_):
        n_freqs = len(freqs_)
        H = np.zeros((N_,len(cfs_)))
        k = freqs_.reshape(-1,1)-cfs_.reshape(1,-1)
        H[:n_freqs] = np.exp((-0.5)*((np.divide(k, sds_))**2))
        H[n_freqs:,:] = np.flip(H[1:int(np.floor((N_+1)/2)),:], axis=0)
        H[0,:] = 0.
        H = np.multiply(H,h_)
        hilbdata = ifft(X_fft[:,np.newaxis] * H, N_, axis=0)
        return hilbdata[:,np.newaxis,:]
    
    # run channels in parallel
    hilb_channels = Parallel(n_jobs=n_jobs)(delayed(_vectorized_band_hilbert)(
        Xf[:,chn], h, N, freqs, cfs, sds) for chn in range(x.shape[1]))
    
    # concatenate channels into a complex-valued 3D array
    hilb_channels = np.concatenate(hilb_channels, axis=1)
    
    return np.angle(hilb_channels), np.abs(hilb_channels), cfs
