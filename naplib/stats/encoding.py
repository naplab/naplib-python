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

    for ii, c in enumerate(np.unique(L)):
        n_c = (L == c).sum()
        mean_diff = (class_means[ii] - overall_mean).reshape(K, 1)
        S_B += n_c * mean_diff @ mean_diff.T
        class_diff = D[L == c] - class_means[ii]
        S_W += class_diff.T @ class_diff

    # Step 1: Calculate Wilks' Lambda
    wilks_lambda = np.linalg.det(S_W) / np.linalg.det(S_B + S_W)

    # Step 2: Compute the F-statistic based on Wilks' Lambda
    numerator_df = (n_classes - 1) * K
    denominator_df = N - n_classes
    F_statistic = ((1 - wilks_lambda ** (1 / K)) / (wilks_lambda ** (1 / K))) * (denominator_df / numerator_df)

    # Step 3: Calculate the p-value using the F-distribution
    p_value = 1 - stats.f.cdf(F_statistic, numerator_df, denominator_df)

    return F_statistic, p_value

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
    allf : np.ndarray
        Allf
    f_stat : float
        F-statistic.
    f_std : float
        Standard dev of allf.

    See Also
    --------
    discriminability
    wilks_lambda_discriminability

    '''
    D = D.T
    
    labels = np.unique(L).squeeze()
        
    Dtot = []
    Ltot = []
    
    for i in range(len(labels)):
        Dtot.append(D[:, np.argwhere(L==labels[i])])
        Ltot.append(L[np.argwhere(L==labels[i])])
        
    D = np.concatenate(Dtot, axis=1).squeeze(-1)
    L = np.concatenate(Ltot, axis=0).squeeze()
    
    
    xh = D.mean(-1).squeeze()
    total = 0.0
        
    
    sbs = []
    sws = []
    
    for cnt1, cnt2 in enumerate(labels):
        # first, between class variability
        index = np.argwhere(L==cnt2)
        index = index.flatten()
        dc = D[:, index].mean(1)
        sb = np.linalg.norm(dc - xh, 2)**2
        
        # now, within class variability
        sw = 0.0
        for cnt3 in index:
            sw += np.linalg.norm(D[:, cnt3] - dc, 2)**2
        
        sbs.append(sb * len(index))
        sws.append(sw)
        total += len(index)
        
    sbs = np.array(sbs)
    sws = np.array(sws)
        
    if np.min(sws) == 0:
        sws[sws==0] = 1e-6
    
    tmpb = sbs / (len(labels)-1)
    tmpw = sws / (total - len(labels))
    allf = np.divide(tmpb, tmpw)
    
    f = tmpb.sum() / tmpw.sum()
    
    fs = np.std(len(labels)*tmpb / tmpw.sum(), ddof=1) / np.sqrt(len(labels))
    
    return allf, f, fs

def discriminability(D, L, elec_mode='all', method='lda-discriminability'):
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
    method : string, default='lda-discriminability'
        Method for computing discriminability. Options are 'lda-discriminability',
        'wilks-lambda'.

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
    if L.ndim > 1:
        ndim_L = L.squeeze(0).ndim
    else:
        ndim_L = L.ndim
    
    label = np.unique(L)

    def _compute_discrim(x_data, labels_data):
        if method == 'lda-discriminability':
            _, f_tmp, _ = lda_discriminability(x_data.T, labels_data)
            return f_tmp, np.nan

        elif method == 'wilks-lambda':
            return wilks_lambda_discriminability(x_data.T, labels_data)
        else:
            raise ValueError(f'Bad method. Must be one of "lda-discriminability", "wilks-lambda"')

    f = []
    if elec_mode == 'all':
        f = np.zeros((D.shape[1]))
        pvalues = np.zeros((D.shape[1]))
        for t in range(D.shape[1]):
            tmp = D[:,t,:].squeeze()
            if ndim_L > 1:
                f_tmp, pvalue = _compute_discrim(tmp, L[t,:].squeeze())
            else:
                f_tmp, pvalue = _compute_discrim(tmp, L.squeeze())
            f[t] = f_tmp
            pvalues[t] = pvalue
        
    elif elec_mode == 'individual':
        f = np.zeros((D.shape[0], D.shape[1]))
        pvalues = np.zeros((D.shape[0], D.shape[1]))
        for t in range(D.shape[1]):
            tmp = D[:,t,:].squeeze()
            for cnt in range(D.shape[0]):
                if ndim_L > 1:
                    f_tmp, pvalue = _compute_discrim(tmp[cnt,:][np.newaxis], L[t,:].squeeze())
                else:
                    f_tmp, pvalue = _compute_discrim(tmp[cnt,:][np.newaxis], L.squeeze())
                f[cnt,t] = f_tmp
                pvalues[cnt,t] = pvalue
        
    else:
        raise Exception(f'Error: elec_mode should be one of ["all", "individual"], but got {elec_mode}')
        
    if method in ['wilks-lambda']:
        return f, pvalues

    return f
    

    
