import warnings
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy import signal as sig
import seaborn as sns


def kde_plot(data, groupings=None, hist=True, alpha=0.2, bins=None, **kwargs):
    """
    Plot kernel density estimate of distribution for data, along with histogram underneath.
    Can plot multiple densities, one per grouping. See Examples below for a depiction.
    
    Parameters
    ----------
    data : list or array-like or list of np.ndarrays
        Data to plot density of. If of shape (N_points,) and groupings is None, then it is assumed to
        be a single distribution to plot. Otherwise, can be either an array of shape (N_points, M_groups),
        or a list of length M_groups containing arrays of shape (N_points_i).
        See ``groupings`` argument for more on how data should be formatted depending on the groupings desired.
    groupings : list or array-like, optional
        Grouping method for separating data into different distributions, or labels for those distributions.
        If groupings is given when data is 1-dimensional, groupings should provide categorical labels
        for each point in data and also be shape (N_points,). Alternatively,
        can be a list or array-like of shape/length M_groups, with each element specifying a
        label for each group/column in ``data``. You must specify groupings for the axis legend to be shown.
    hist : bool, default=True
        If True (default), plots a histogram underneath the kernel density estimate.
    alpha : float, default=0.2
        Alpha value for transparency of histogram. Ignored if ``hist=False``.
    bins : int or sequence or str, default = :rc:`hist.bins`
        Bins for histogram. Ignored if ``hist=False``.
    **kwargs : kwargs
        kwargs for seaborn.kdeplot. Cannot include 'data', 'x', 'y', or 'hue'. See below for
        some examples of frequently used kwargs.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    bw_method : string, scalar, or callable, optional
        Method for determining the smoothing bandwidth to use; passed to scipy.stats.gaussian_kde.
        Can be a single float to determine the bandwidth.
    color : str or matplotlib color, or list of colors, optional
        Color to use if only providing 1 grouping of data (e.g. if ``groupings=None``), or an iterable
        of the color to use for each group.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        matplotlib axes containing the plot
        
    Examples
    --------
    >>> from naplib.visualization import kde_plot
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> rng = np.random.default_rng(1)
    >>> data = rng.normal(size=(100,))
    >>> data[50:] += 0.5 # shift the second half of the samples
    >>> groupings = np.array(['G0'] * 100) # define grouping vector
    >>> groupings[50:] = 'G1' # set a different label for the samples we shifted
    >>> # plot the density for each group, as well as a histogram underneath each
    >>> ax = kde_plot(data, groupings=groupings, bw_method=0.25, bins=15, color=['k','r'])

    .. figure:: /figures/kdeplot1.png
        :width: 400px
        :alt: kde_plot figure
        :align: center

    >>> # plot the exact same figure from a list of arrays and grouping labels of same length
    >>> data_list = [data[:50],data[50:]]
    >>> kde_plot(data_list, groupings=['G0','G1'], bw_method=0.25, bins=15, color=['k','r'])
    >>> # plot the exact same figure from a 2D numpy array
    >>> data_mat = np.concatenate([data[:50,np.newaxis],data[50:,np.newaxis]], axis=1)
    >>> kde_plot(data_mat, groupings=['G0','G1'], bw_method=0.25, bins=15, color=['k','r'])
    >>> # if we don't pass in groupings but data is still a 2D array or a list,
    >>> # then there just won't be a legend, but the plot will be the same
    >>> kde_plot(data_mat, bw_method=0.25, bins=15, color=['k','r'])

    """
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()
    
    if isinstance(data, list) and all([not isinstance(xx, np.ndarray) for xx in data]):
        data = np.asarray(data)
        
    original_groupings_none = groupings is None
    
    if isinstance(data, np.ndarray):
        
        if data.ndim == 1 or data.shape[1]==1:
            if groupings is None:
                groupings2 = np.zeros_like(data).astype('int') # all one group
            else:
                groupings2 = [str(x) for x in groupings]
            if len(groupings2) != len(data):
                raise ValueError(f'data and groupings must be same length, but got data'
                                 f' with length {len(data)} and groupings with length {len(groupings)}')
            df = pd.DataFrame.from_dict({'data': data, 'group': groupings2})
            
        else: # multiple things to plot since ndim>1
            assert data.ndim > 1
            if groupings is None:
                groupings = np.zeros_like(data) + np.arange(data.shape[1]) # array of shape (N_points, M_groups)
                groupings2 = groupings.flatten('F').astype('int') # e.g. now [0,0,0,1,1,1,2,2,2]
            elif len(groupings) == data.shape[1]:
                groupings2 = []
                for g in groupings:
                    groupings2 += [g] * len(data)
            else:
                raise TypeError(f'Invalid format for groupings when data is multidimensional numpy array.'
                                f' Must be a list of length data.shape[1]')
                
            df = pd.DataFrame.from_dict({'data': data.flatten('F'), 'group': groupings2})
            
    elif isinstance(data, list):
        if not all([isinstance(xx, np.ndarray) for xx in data]):
            raise TypeError(f'If data is a list, each element must be a numpy array')
        
        if groupings is None:
            groupings = [int(i) for i in range(len(data))]
        if len(data) != len(groupings):
            raise ValueError(f'groupings must be same length as data if data is given as list, '
                             f'but got data with length {len(data)}, groupings with length {len(groupings)}')
        
        groupings_flat = []
        for ii, d in enumerate(data):
            for _ in d:
                groupings_flat.append(groupings[ii])
        
        df = pd.DataFrame.from_dict({'data': np.concatenate(data, axis=0), 'group': groupings_flat})
        
    else:
        raise TypeError(f'data must be either a np.ndarray, a list of scalars, or a list of np.ndarray, but got {type(data)}')
        
    xlbl_before = ax.xaxis.get_label().get_text()
    ylbl_before = ax.yaxis.get_label().get_text()
    
    color = None
    if 'color' in kwargs:
        color = kwargs.pop('color')
    if color is None:
        color = [None for _ in np.unique(df['group'].values)]
    elif color is not None and not isinstance(color, list):
        color = [color]
        
    if len(color) != len(np.unique(df['group'].values)):
        num_unique = len(np.unique(df['group'].values))
        raise ValueError(f'If specified, number of colors provided must match number of groups,'
                         f' but got {len(color)} colors and {num_unique} groups')

    # loop through groups
    for i, grp in enumerate(sorted(np.unique(df['group'].values))):
        if color[i] is None:
            col = next(ax._get_lines.prop_cycler)['color']
        else:
            col = color[i]

        this_group_df = df.loc[df['group']==grp]
        sns.kdeplot(data=this_group_df, ax=ax, x='data', color=col, label=grp, **kwargs)
        # add histogram
        if hist:
            ax.hist(this_group_df['data'], bins=bins, color=col, density=True, alpha=alpha)
        
    if not original_groupings_none:
        ax.legend()
    
    ax.set_xlabel(xlbl_before)
    if ylbl_before == '':
        ax.set_ylabel(ylbl_before)

    return ax


def shaded_error_plot(*args, ax=None, reduction='mean', err_method='stderr', color=None, alpha=0.4, plt_args={}, shade_args={}, nan_policy='omit'):
    '''
    Plot the average/median value at each time point and a shaded region indicating error or confidence
    level above and below the line. See Examples below for a depiction.

    Parameters
    ----------
    x : array-like, shape (n_samples,), optional
        *x* values are optional and default to ``range(len(y))``.
    y : array-like, shape (n_samples, n_points)
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
    reduction : str, default='mean'
        Reduction method, either 'mean' or 'median'.
    err_method : string or float, default='stderr'
        The method to use to calculate error bars. If a string, one of ['stderr','std'].
        If a float, defines the confidence interval desired. For example 0.95 specifies
        a 95% confidence interval around the mean (i.e. the interval from the 2.5th percentile
        to the 97.5th percentile). Note, if the data have significant outliers and reduction='mean'
        then the confidence interval bounds might not surround the mean value line.
    color : str, default=None
        Color to plot line and shaded region. Defaults to next color in color cycle.
    alpha : float, default=0.4
        Shading alpha. Value between 0 and 1.
    plt_args : dict, default={}
        Dict of args to be passed to plt.plot(). e.g. {'linewidth': 2}, etc.
    shade_args : dict, default={}
        Dict of args to be passed to plt.fill_between(). e.g. {'alpha': 0.2}, etc.
    nan_policy : string, default='omit'
        One of ['omit','raise','propogate']. If 'omit', will ignore any nan in the
        inputs, if 'raise', will raise a ValueError if nan is found in input, if
        'propogate', do not do anything special with nan values.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        matplotlib axes containing the plot

    Examples
    --------
    >>> from naplib.visualization import shaded_error_plot as sep
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> x, y = np.linspace(0, 1, 10), rng.normal(size=(10,5))
    >>> fig, ax = plt.subplots(3,1)
    >>> sep(y, ax=axes[0]) # plot mean of y, with shaded error regions
    >>> sep(y, 'r--', ax=axes[1]) # same plot but color is red and line is dashed
    >>> sep(x, y, ax=axes[2], err_method='std') # plot vs specific x values and use std. error
    >>> plt.show()

    .. figure:: /figures/shadederrorplot1.png
        :width: 400px
        :alt: shaded_error_plot figure
        :align: center
    
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
        
    if reduction == 'mean':
        if nan_policy == 'omit':
            reduction_func = np.nanmean
        else:
            reduction_func = np.mean
    elif reduction == 'median':
        if nan_policy == 'omit':
            reduction_func = np.nanmedian
        else:
            reduction_func = np.median
    else:
        raise ValueError(f'reduction must be either "mean" or "median", but got {reduction}')
            
    allowed_errors = ['stderr','std']
    if isinstance(err_method, str):
        if err_method not in allowed_errors:
            raise ValueError(f'err_method is a string but is not one of {allowed_errors}, but rather {err_method}')
    elif not isinstance(err_method, float):
        raise ValueError(f'err_method must be either a string or a float, but got {err_method}')
    elif isinstance(err_method, float) and (err_method <= 0 or err_method > 1.0):
        raise ValueError(f'If err_method is a float then it must be in the range (0, 1]')
    if nan_policy == 'omit':
        y_mean = reduction_func(y, axis=1)
        if err_method == 'stderr':
            y_err = np.nanstd(y, axis=1) / np.sqrt(y.shape[1])
        elif err_method == 'std':
            y_err = np.nanstd(y, axis=1)
        else:
            alpha_level = 1.0 - err_method
            y_err = [np.nanpercentile(y, 100*alpha_level/2., axis=1), np.nanpercentile(y, 100*(1-(alpha_level/2.)), axis=1)]
    else: # propogate, since 'raise' has already been taken care of
        y_mean = reduction_func(y, axis=1)
        if err_method == 'stderr':
            y_err = y.std(1) / np.sqrt(y.shape[1])
        elif err_method == 'std':
            y_err = y.std(1)
        else:
            alpha_level = 1.0 - err_method
            y_err = [np.percentile(y, 10*alpha_level/2., axis=1), np.percentile(y, 10*(1-(alpha_level/2.)), axis=1)]
    
    if fmt == '':
        line_, = ax.plot(x, y_mean, **plt_args)
    else:
        line_, = ax.plot(x, y_mean, fmt, **plt_args)
                     
    color = line_.get_color()
    shade_args['color'] = color
    if isinstance(err_method, str):
        ax.fill_between(x, y_mean-y_err, y_mean+y_err, **shade_args)
    else:
        ax.fill_between(x, y_err[0], y_err[1], **shade_args)

    return ax

def hierarchical_cluster_plot(data, axes=None, varnames=None, cmap='bwr', n_clusters=2):
    '''
    Perform hierarchical clustering and plot dendrogram and clustered values as an
    image underneath. See Examples below for a depiction.

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
    axes : np.ndarray of matplotlib.axes.Axes
        Axes where data was plotted. Only returned if axes were not passed in.

    Examples
    --------
    >>> from naplib.visualization import hierarchical_cluster_plot as hcp
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng(10)
    >>> x = rng.normal(size=(100,5))
    >>> x[:,1] += rng.normal(loc=1, scale=3, size=(100,))
    >>> x[:,2] += rng.normal(loc=-1, scale=3, size=(100,))
    >>> varnames = ['var1','var2','var3','var4','var5']
    >>> clust, labels, fig, axes = hcp(x, varnames=varnames)

    .. figure:: /figures/hierarchicalclusterplot1.png
        :width: 400px
        :alt: hierarchical_cluster_plot figure
        :align: center

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


def strf_plot(coef, tmin=None, tmax=None, freqs=None, ax=None, smooth=True, vmax=None):
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
    vmax : float, optional
        If provided, colormap will be between [-vmax, vmax]. If not given,
        uses the max absolute value of the coef.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes where STRF coef is plotted.

    Examples
    --------
    >>> from naplib.visualization import strf_plot
    >>> import numpy as np
    >>> from scipy.stats import multivariate_normal
    >>> # generate example STRF weights following mne's example:
    >>> # https://mne.tools/stable/auto_tutorials/machine-learning/30_strf.html 
    >>> fs = 100
    >>> n_freqs = 32
    >>> tmin, tmax = 0, 0.4
    >>> delays_samp = np.arange(np.round(tmin * fs),
    ...                         np.round(tmax * fs) + 1).astype(int)
    >>> delays_sec = delays_samp / fs
    >>> freqs = np.linspace(50, 5000, n_freqs)
    >>> grid = np.array(np.meshgrid(delays_sec, freqs))
    >>> # We need data to be shaped as n_epochs, n_features, n_times, so swap axes here
    >>> grid = grid.swapaxes(0, -1).swapaxes(0, 1)
    >>> # Simulate a temporal receptive field with a Gabor filter
    >>> means_high = [.1, 500]
    >>> means_low = [.2, 2500]
    >>> cov = [[.001, 0], [0, 500000]]
    >>> gauss_high = multivariate_normal.pdf(grid, means_high, cov)
    >>> gauss_low = -1 * multivariate_normal.pdf(grid, means_low, cov)
    >>> weights = gauss_high + gauss_low  # Combine to create the "true" STRF
    >>> strf_plot(weights, tmin=tmin, tmax=tmax)

    .. figure:: /figures/imSTRF1.png
        :width: 400px
        :alt: strf_plot figure
        :align: center

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
        
    if vmax is not None:
        kwargs['vmin'] = -vmax
        kwargs['vmax'] = vmax
        
    ax.pcolormesh(delays_sec, freqs_, coef, **kwargs)
    
    if freqs is not None:
        ax.set_yticks([0, coef.shape[0]-1])
        ax.set_yticklabels([freqs[0], freqs[-1]])

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

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes where STRF coef is plotted.

    Examples
    --------
    >>> import naplib as nl
    >>> from naplib.visualization import freq_response
    >>> from naplib.preprocessing import filter_butter
    >>> # Load sample data to filter
    >>> data = nl.io.load_speech_task_data()
    >>> alpha_band_data, filters = filter_butter(data, btype='bandpass',
    ...                                          Wn=[10, 20],
    ...                                          return_filters=True)
    >>> ax = freq_response(filters[0], fs=data[0]['dataf'])

    .. figure:: /figures/freq_responses1.png
        :width: 400px
        :alt: frequency response figure
        :align: center

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
        
    ax.set_title('Frequency Response')
    if units == 'Hz':
        ax.set_xlabel('Frequency (Hz)')
    else:
        ax.set_xlabel('Frequency (radians / second)')
    ax.set_ylabel('Amplitude (dB)')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    return ax
    
