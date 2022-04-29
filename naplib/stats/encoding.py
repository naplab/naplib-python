import numpy as np

def dprime(D, L):
    '''
    Compute DPrime.
    
    Parameters
    ----------
    D : array-like of shape (features, instance)
        Data features.
    L : array-like of shape (instance, )
        Labels for each instance.ß
    
    Returns
    -------
    allf : np.ndarray
        Allf
    f_stat : float
        F-statistic.
    f_std : float
        Standard dev of allf.

    '''
    
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
        
def fratio(D, L, elec_mode='all'):
    '''
    F ratio over time.
    
    Parameters
    ----------
    D : array-like of shape (electrodes, time, instances)
        Data features over time.
    L : array-like containing labels for each instance
        L is either of shape (1, instances) if each instance has the same 
        label across the full time axis, or is of shape (time, instances)
    elec_mode : string, one of ['all', 'individual']
        if 'all', computes f-ratio over all electrodes together, and returns
        f-ratio of shape (time,)
        if 'individual', computes f-ratio for each electrode individually
        and returns f-ratio of shape (electrodes, time)
        
    Returns
    -------
    fratio : np.ndarray
        if elec_mode=='all', shape=(time,)
        if elec_mode=='individual', shape=(electrodes, time)
    
    '''
    
    ndim_L = L.squeeze().ndim
    
    label = np.unique(L)
    
    f = []
    if elec_mode == 'all':
        f = np.zeros((D.shape[1]))
        for t in range(D.shape[1]):
            tmp = D[:,t,:].squeeze()
            if ndim_L > 1:
                _, f_tmp, _ = dprime(tmp, L[t,:].squeeze())
            else:
                _, f_tmp, _ = dprime(tmp.reshape(tmp.shape[0], -1), L.squeeze())
            f[t] = f_tmp
        
    elif elec_mode == 'individual':
        f = np.zeros((D.shape[0], D.shape[1]))
        for t in range(D.shape[1]):
            tmp = D[:,t,:].squeeze()
            for cnt in range(D.shape[0]):
                if ndim_L > 1:
                    _, f_tmp, _ = dprime(tmp[cnt,:], L[t,:].squeeze())
                else:
                    _, f_tmp, _ = dprime(tmp[cnt,:].reshape(1,-1), L.squeeze())
                f[cnt,t] = f_tmp
        
    else:
        raise Exception(f'Error: elec_mode should be one of ["all", "individual"], but got {elec_mode}')
        
    return f
    
    
