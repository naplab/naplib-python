import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def shadederrorplot(x, y, ax=None, err_method='stderr', plt_args={}, shade_args={}, nan_policy='omit'):
    '''
    Parameters
    ----------
    x : shape (time,)
    y : shape (time, n_samples)
    ax : plt axes to use
    err_method : string, default='stderr
        One of ['stderr','std'], the method to use to calculate error bars.
    plt_args : dict
        Args to be passed to plt.plot(). e.g. 'color','linewidth',...
    shade_args : dict
        Args to be passed to plt.fill_between(). e.g. 'color','alpha',...
    nan_policy : string, default='omit'
        One of ['omit','raise','propogate']. If 'omit', will ignore any nan in the
        inputs, if 'raise', will raise a ValueError if nan is found in input, if
        'propogate', do not do anything special with nan values.
        
    Raises
    ------
    ValueError
        if nan found in input
    '''
    if nan_policy not in ['omit','raise','propogate']:
        raise Exception(f"nan_policy must be one of ['omit','raise','propogate'], but found {nan_policy}")
    if nan_policy == 'raise':
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError('Found nan in input')
    if ax is None:
        ax = plt.gca()
    if 'alpha' not in shade_args:
        shade_args['alpha'] = 0.5

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
        
    ax.plot(x, y_mean, **plt_args)
    ax.fill_between(x, y_mean-y_err, y_mean+y_err, **shade_args)
    

def hierarchicalclusterplot(data, axes=None, varnames=None, cmap='bwr', n_clusters=2):
    '''
    Parameters
    ----------
    data : shape (n_samples, n_features)
    axes : array of length 2 containing matplotlib axes (optional)
        axes[0] will be for the dendrogram and axes[1] will be for the data
    varnames : list of strings, length must = n_features, default=None
        variable names which will be printed as yticklabels on the data plot
    cmap : string, default='bwr'
        colormap for the data plot
    n_clusters : int, default=2
        number of clusters which will be used when computing cluster labels that are returned
    
    Returns
    -------
    cluster_dict : dict output from scipy.cluster.hierarchy.dendrogram
    cluster_labels : np.array
        cluster labels from sklearn.cluster.AgglomerativeClustering
    '''
    if axes is None:
        _, axes = plt.subplots(2,1,figsize=(10, 7), gridspec_kw={'height_ratios': [2.5,1]})
        
    dend = shc.dendrogram(shc.linkage(data, method='ward'), show_leaf_counts=False, ax=axes[0], get_leaves=True, no_labels=True)

    axes[0].set_yticks([])

    leaves = dend['leaves']

    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster_labels = cluster.fit_predict(data)

    if cmap=='bwr':
        mm1 = np.abs(data.reshape((-1)).min())
        mm2 = np.abs(data.reshape((-1)).max())
        mm = max([mm1, mm2])
        axes[1].imshow(data[leaves,:].T, cmap=cmap, aspect=4, vmin=-mm, vmax=mm, interpolation='none')
    else:
        axes[1].imshow(data[leaves,:].T, cmap=cmap, aspect=4, interpolation='none')
    if varnames:
        axes[1].set_yticks([i for i in range(len(varnames))])
        axes[1].set_yticklabels(varnames, fontsize=8)

    axes[1].set_xticks([])

    plt.tight_layout()
    plt.show()
    
    return dend, cluster_labels


def imSTRF(coef, tmin=None, tmax=None, freqs=None, ax=None, smooth=True):
    '''
    Plot STRF weights as image. Weights are automatically centered at 0
    so the center of the colormap is 0.
    
    Parameters
    ----------
    coef : np.array, shape (freq, lag)
        STRF weights.
    tmin : float, optional
        Time of first lag (first column)
    tmax : float, optional
        Time of final lag (last column)
    freqs : list or array-like, length=2, optional
        Frequency of lowest and highest frequency bin in STRF.
    ax : plt.Axes, optional
        Axes to plot on.
    smooth : bool, default=True
        Whether or not to smooth the STRF image. Smoothing is
        done with 'gouraud' shading in plt.pcolormesh().
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
