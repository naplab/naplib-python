import numpy as np
import scipy.stats as stats


def wilks_lambda_discriminability(D, L):
    '''
    Compute Wilks' Lambda F-ratio and p-value for a single data matrix (not over time).
    
    Parameters
    ----------
    D : array-like of shape (instance, features)
        Data features.
    L : array-like of shape (instance, )
        Labels for each instance.
    
    Returns
    -------
    f_stat : float
        F-statistic
    p_val : float
        p-value

    See Also
    --------
    discriminability
    lda_discriminability

    '''
    N, K = D.shape
    n_classes = len(np.unique(L))

    # Compute the overall mean of X and the class means for each category in Y
    overall_mean = D.mean(axis=0)
    class_means = [D[L == c].mean(axis=0) for c in np.unique(L)]

    # Compute the between-class scatter matrix and the within-class scatter matrix
    S_B = np.zeros((K, K))
    S_W = np.zeros((K, K))

    for i, c in enumerate(np.unique(L)):
        n_c = (L == c).sum()
        mean_diff = (class_means[i] - overall_mean).reshape(K, 1)
        S_B += n_c * mean_diff @ mean_diff.T
        class_diff = D[L == c] - class_means[i]
        S_W += class_diff.T @ class_diff

    # Step 1: Calculate Wilks' Lambda
    wilks_lambda = np.linalg.det(S_W) / np.linalg.det(S_B + S_W)

    # Step 2: Compute the F-statistic based on Wilks' Lambda
    numerator_df = (n_classes - 1) * K
    denominator_df = N - n_classes
    f_stat = (
        (1 - wilks_lambda ** (1 / K)) / (wilks_lambda ** (1 / K))
        * (denominator_df / numerator_df)
    )

    # Step 3: Calculate the p-value using the F-distribution
    p_value = 1 - stats.f.cdf(f_stat, numerator_df, denominator_df)

    return f_stat, p_value


def lda_discriminability(D, L):
    '''
    Compute LDA discriminability for a single data matrix (not over time).
    
    Parameters
    ----------
    D : array-like of shape (instance, features)
        Data features.
    L : array-like of shape (instance, )
        Labels for each instance.
    
    Returns
    -------
    f_all : np.ndarray
        All F-statistics.
    f_stat : float
        F-statistic.
    f_std : float
        Standard dev of f_all.

    See Also
    --------
    discriminability
    wilks_lambda_discriminability

    '''
    labels = list(np.unique(L))
    def where_label_is(label):
        return np.argwhere(L == label).squeeze(-1)
        
    D = np.concatenate([D[where_label_is(label)] for label in labels])
    L = np.concatenate([L[where_label_is(label)] for label in labels])
    
    global_mean = D.mean(0)
    
    sbs = np.zeros(len(labels))
    sws = np.zeros(len(labels))
    total = 0.0
    for i, label in enumerate(labels):
        index = where_label_is(label)

        # Between class variability
        group_mean = D[index].mean(0)
        sbs[i] = np.linalg.norm(group_mean - global_mean, 2)**2 * len(index)
        
        # Within class variability
        sws[i] = sum(np.linalg.norm(D[j] - group_mean, 2)**2 for j in index)
        
        total += len(index)
    
    # Set zeros to epsilon for numerical reasons
    sws[sws == 0] = 1e-6
    
    # Normalize
    sbs /= (len(labels) - 1)
    sws /= (total - len(labels))

    f_all = np.divide(sbs, sws)
    f_stat = sbs.sum() / sws.sum()
    f_std = np.std(len(labels)*sbs / sws.sum(), ddof=1) / np.sqrt(len(labels))

    return f_all, f_stat, f_std


def discriminability(D, L, elec_mode='all', method='lda'):
    '''
    Compute discriminability of classes over time. Can use multiple electrodes jointly,
    or compute the discriminability by each electrode individually. Multiple methods
    available.
    
    Parameters
    ----------
    D : array-like of shape (electrodes, time, instances)
        Data features over time.
    L : array-like containing labels for each instance
        L is either of shape (instances,) if each instance has the same 
        label across the full time axis, or is of shape (time, instances). Must be
        categorical to indicate class membership for each instance.
    elec_mode : string, one of ['all', 'individual']
        if 'all', computes f-ratio over all electrodes together, and returns
        f-ratio of shape (time,)
        if 'individual', computes f-ratio for each electrode individually
        and returns f-ratio of shape (electrodes, time)
    method : string, default='lda'
        Method for computing discriminability. Options are 'lda', 'wilks-lambda'.

    Returns
    -------
    fratio : np.ndarray
        if elec_mode=='all', shape=(time,)
        if elec_mode=='individual', shape=(electrodes, time)
    pvalues : np.ndarray
        Only returned if method is 'wilks-lambda'. Shape is the same
        as ``fratio``.

    Examples
    --------
    >>> from naplib.stats import discriminability
    >>> rng = np.random.default_rng(1)
    >>> elecs, t, instances = 2, 5, 50
    >>> D = np.concatenate([rng.normal(size=(elecs, t, instances)),
    ...                     rng.normal(loc=1, scale=0.5, size=(elecs, t, instances))],
    ...                    axis=-1)
    >>> # labels for the data, where labels do not change over time
    >>> L = np.concatenate([np.zeros((instances,)), np.ones((instances,))])
    >>> f_stat, p_val = discriminability(D, L, method='wilks-lambda')
    >>> f_stat
    array([16.71955996, 19.94997217, 19.0641678 , 15.95256107, 17.90728111])
    >>> p_val
    array([5.65679222e-07, 5.38811997e-08, 1.01531199e-07, 1.00551882e-06, 2.35186401e-07]))

    '''
    def _compute_discrim(x_data, labels_data):
        if method == 'lda':
            _, f_stat, _ = lda_discriminability(x_data.T, labels_data)
            return f_stat, np.nan
        elif method == 'wilks-lambda':
            return wilks_lambda_discriminability(x_data.T, labels_data)
        else:
            raise ValueError('Bad method. Must be one of "lda", "wilks-lambda"')

    f_stat, p_vals  = None, None
    if elec_mode == 'all':
        f_stat = np.zeros(D.shape[1])
        p_vals = np.zeros(D.shape[1])
        for t in range(D.shape[1]):
            if L.ndim > 1:
                f_stat[t], p_vals[t] = _compute_discrim(D[:,t], L[t])
            else:
                f_stat[t], p_vals[t] = _compute_discrim(D[:,t], L)
    elif elec_mode == 'individual':
        f_stat = np.zeros(D.shape[:2])
        p_vals = np.zeros(D.shape[:2])
        for t in range(D.shape[1]):
            for e in range(D.shape[0]):
                if L.ndim > 1:
                    f_stat[e,t], p_vals[e,t] = _compute_discrim(D[e,t,None], L[t])
                else:
                    f_stat[e,t], p_vals[e,t] = _compute_discrim(D[e,t,None], L)
    else:
        raise Exception(
            f'Error: elec_mode should be one of ["all", "individual"], but got {elec_mode}'
        )
    
    if method in ['wilks-lambda']:
        return f_stat, p_vals
    
    return f_stat

