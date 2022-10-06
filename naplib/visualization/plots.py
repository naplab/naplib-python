import warnings
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy import signal as sig


def shadederrorplot(*args, ax=None, err_method='stderr', color=None, alpha=0.4, plt_args={}, shade_args={}, nan_policy='omit'):
    '''
    Parameters
    ----------
    x : array-like, shape (n_samples,), optional
        *x* values are optional and default to ``range(len(y))``.
    y : array-like, shape (n_samples, n_lines)
        Data to plot, providing the vertical coordinates. *y* values should be
        two-dimensional, and statistics used to compute shaded region interval
        are computed over the second dimension.
    fmt : str, optional
        A format string, e.g. 'ro' for red circles. See the matplotlib
        `Axes.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_
        Notes section for a full description of the format strings.
        Format strings are just an abbreviation for quickly setting
        basic line properties. All of these and more can also be
        controlled by keyword arguments within color or plt_args.
        This argument cannot be passed as keyword.
    ax : plt.Axes instance, optional
        Axes to use. If not specified, will use current axes.
    color : str, default=None
        Color to plot line and shaded region. Defaults to next color in color cycle.
    alpha : float, default=0.4
        Shading alpha. Value between 0 and 1.
    err_method : string, default='stderr
        One of ['stderr','std'], the method to use to calculate error bars.
    plt_args : dict, default={}
        Dict of args to be passed to plt.plot(). e.g. {'linewidth': 2}, etc.
    shade_args : dict, default={}
        Dict of args to be passed to plt.fill_between(). e.g. {'alpha': 0.2}, etc.
    nan_policy : string, default='omit'
        One of ['omit','raise','propogate']. If 'omit', will ignore any nan in the
        inputs, if 'raise', will raise a ValueError if nan is found in input, if
        'propogate', do not do anything special with nan values.
    
    Examples
    --------
    >>> from naplib.visualization import shadederrorplot as sep
    >>> import matplotlib.pyplot as plt
    >>> x, y = np.linspace(0, 1, 10), np.random.rand(10,5)
    >>> fig, ax = plt.subplots()
    >>> sep(y) # plot mean of y vs x, with shaded error regions
    >>> sep(y, 'r--') # same plot but color is red and line is dashed
    >>> sep(x, y) # same plot but against specific x values
    >>> plt.show()
    
    Raises
    ------
    ValueError
        if nan found in input and ``nan_policy`` is 'raise'.
    '''
    
    if color is not None:
        plt_args['color'] = color

    shade_args['alpha'] = alpha
    
    fmt = ''
    x = None
    y = None
    if len(args) == 0:
        raise ValueError('No data provided to plot.')
    elif len(args) == 1:
        y = args[0]
    elif len(args) == 2:
        if isinstance(args[1], str):
            y, fmt = args
        else:
            x, y = args
    elif len(args) == 3:
        x, y, fmt = args
    else:
        raise ValueError(f'Too many args passed. Expected at most 3 (x, y, fmt)')
    
    if not isinstance(fmt, str):
        raise TypeError(f'fmt must be of type string but got {type(fmt)}')
    if x is None:
        x = np.arange(len(y))
    
    if y.ndim == 1:
        y = y[:,np.newaxis]

    if nan_policy not in ['omit','raise','propogate']:
        raise Exception(f"nan_policy must be one of ['omit','raise','propogate'], but found {nan_policy}")
    if nan_policy == 'raise':
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError('Found nan in input')

    if ax is None:
        ax = plt.gca()
            
    allowed_errors = ['stderr','std']
    if err_method not in allowed_errors:
        raise ValueError(f'err_method must be one of {allowed_errors}, but found {err_method}')
    if nan_policy == 'omit':
        y_mean = np.nanmean(y, axis=1)
        if err_method == 'stderr':
            y_err = np.nanstd(y, axis=1) / np.sqrt(y.shape[1])
        elif err_method == 'std':
            y_err = np.nanstd(y, axis=1)
    else:
        y_mean = y.mean(1)
        if err_method == 'stderr':
            y_err = y.std(1) / np.sqrt(y.shape[1])
        elif err_method == 'std':
            y_err = y.std(1)
        
    
    if fmt == '':
        line_, = ax.plot(x, y_mean, **plt_args)
    else:
        line_, = ax.plot(x, y_mean, fmt, **plt_args)
                     
    color = line_.get_color()
    shade_args['color'] = color
    ax.fill_between(x, y_mean-y_err, y_mean+y_err, **shade_args)
    

def hierarchicalclusterplot(data, axes=None, varnames=None, cmap='bwr', n_clusters=2):
    '''
    Parameters
    ----------
    data : shape (n_samples, n_features)
        Data to cluster and display. 
    axes : list of plt.Axes, length 2, optional
        array of length 2 containing matplotlib axes to plot on.
        axes[0] will be for the dendrogram and axes[1] will be for the data. If not
        specified, will create new axes in subplots.
    varnames : list of strings, length must = n_features, default=None
        variable names which will be printed as yticklabels on the data plot
    cmap : string, default='bwr'
        colormap for the data plot
    n_clusters : int, default=2
        number of clusters which will be used when computing cluster labels that are returned
    
    Returns
    -------
    cluster_dict : dict
        output from scipy.cluster.hierarchy.dendrogram
    cluster_labels : np.ndarray
        cluster labels from sklearn.cluster.AgglomerativeClustering, shape=(n_samples,)
    fig : matplotlib figure
        Figure where data was plotted. Only returned if axes were not passed in.
    axes : array of Axes
        Axes where data was plotted. Only returned if axes were not passed in.
    '''
    if axes is None:
        return_axes = True
        fig, axes = plt.subplots(2,1,figsize=(10, 7), gridspec_kw={'height_ratios': [1,2]})
    else:
        return_axes = False
        
    dend = shc.dendrogram(shc.linkage(data, method='ward'), show_leaf_counts=False, ax=axes[0], get_leaves=True, no_labels=True)

    axes[0].set_yticks([])

    leaves = dend['leaves']

    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster_labels = cluster.fit_predict(data)

    if cmap=='bwr':
        mm1 = np.abs(data.reshape((-1)).min())
        mm2 = np.abs(data.reshape((-1)).max())
        mm = max([mm1, mm2])
        axes[1].imshow(data[leaves,:].T, cmap=cmap, aspect='auto', vmin=-mm, vmax=mm, interpolation='none')
    else:
        axes[1].imshow(data[leaves,:].T, cmap=cmap, aspect='auto', interpolation='none')
    if varnames:
        axes[1].set_yticks([i for i in range(len(varnames))])
        axes[1].set_yticklabels(varnames, fontsize=8)

    axes[1].set_xticks([])
    
    if return_axes:
        return dend, cluster_labels, fig, axes

    return dend, cluster_labels


def imSTRF(coef, tmin=None, tmax=None, freqs=None, ax=None, smooth=True, return_ax=False):
    '''
    Plot STRF weights as image. Colormap is automatically centered at 0 so
    that 0 corresponds to white, positive values are red, and negative values
    are blue.
    
    Parameters
    ----------
    coef : np.array, shape (freq, lag)
        STRF weights.
    tmin : float, optional
        Time of first lag (first column in coef)
    tmax : float, optional
        Time of final lag (last column in coef)
    freqs : list or array-like, length=2, optional
        Frequency of lowest and highest frequency bin in STRF.
    ax : plt.Axes, optional
        Axes to plot on. If not specified, will use current axes.
    smooth : bool, default=True
        Whether or not to smooth the STRF image. Smoothing is
        done with 'gouraud' shading in plt.pcolormesh().
    return_ax : bool, default=False
        Whether or not to return axes as well.

    Returns
    -------
    ax : matplotlib Axes
        Axes where STRF coef is plotted. Only returned if ``return_ax`` is True.
    '''
    
    if ax is None:
        ax = plt.gca()
    
    if tmin is not None and tmax is not None:
        delays_sec = np.linspace(tmin, tmax, coef.shape[1])
        lag_string = 'Lag (s)'
    else:
        delays_sec = np.arange(0, coef.shape[1])
        lag_string = 'Lag (samples)'
        
    ax.set_xlabel(lag_string)
    
    freqs_ = np.arange(0, coef.shape[0])
    ax.set_ylabel('Frequency')
        
    if smooth:
        kwargs = dict(vmax=np.abs(coef).max(), vmin=-np.abs(coef).max(),
                  cmap='bwr', shading='gouraud')
    else:
        kwargs = dict(vmax=np.abs(coef).max(), vmin=-np.abs(coef).max(),
                  cmap='bwr')
        
    ax.pcolormesh(delays_sec, freqs_, coef, **kwargs)
    
    if freqs is not None:
        yticks = ax.get_yticks()
        ax.set_yticks([0, coef.shape[0]-1])
        ax.set_yticklabels([freqs[0], freqs[-1]])

    if return_ax:
        return ax


def freq_response(ba, fs, ax=None, units='Hz'):
    '''
    Plot frequency response of a digital filter.
    
    Parameters
    ----------
    ba : tuple of length 2
        Tuple containing (b, a), the filter numerator and denominator polynomials.
    fs : int
        Sampling rate in Hz.
    ax : plt.Axes instance, optional
        Axes to use. If not specified, will use current axes.
    units : string
        One of {'Hz', 'rad/s'} specifying whether to plot frequencies in Hz or
        radians per second.
    '''
    if units not in ['Hz','rad/s']:
        raise ValueError(f'units must be one of ["Hz", "rad/s"] but got {units}')
        
    if ax is None:
        ax = plt.gca()
        
    if units == 'Hz':
        w, h = sig.freqz(ba[0], ba[1], fs=fs)
    else:
        w, h = sig.freqs(ba[0], ba[1])
        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
        ax.semilogx(w, 20 * np.log10(abs(h)))
        
    ax.set_title('Butterworth filter frequency response')
    if units == 'Hz':
        ax.set_xlabel('Frequency (Hz)')
    else:
        ax.set_xlabel('Frequency (radians / second)')
    ax.set_ylabel('Amplitude (dB)')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')
    
