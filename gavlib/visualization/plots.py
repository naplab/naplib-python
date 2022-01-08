import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def shadederrorplot(x, y, ax=None, plt_args={}, shade_args={}):
    '''
    Inputs
    ------
    x : shape (time,)
    y : shape (time, n_samples) 
    '''
    if ax is None:
        ax = plt.gca()
    y_mean = y.mean(1)
    y_err = y.std(1) / np.sqrt(y.shape[1])
    ax.plot(x, y_mean, **plt_args)
    ax.fill_between(x, y_mean-y_err, y_mean+y_err, **shade_args)
    

def hierarchicalclusterplot(data, axes=None, varnames=None, cmap='bwr', n_clusters=2):
    '''
    Inputs
    ------
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

    axes[1].imshow(data[leaves,:].T, cmap=cmap, aspect=4)
    if varnames:
        axes[1].set_yticks([i for i in range(len(varnames))])
        axes[1].set_yticklabels(varnames, fontsize=8)
    else:
        axes[1].set_yticks([])

    axes[1].set_xticks([])

    plt.tight_layout()
    plt.show()
    
    return dend, cluster_labels