import numpy as np

def get_stars_for_pvalues(pvals):
    '''
    Convert pvalues to strings of stars for significance.
    thresholds: [inf, 0.05, 0.01, 0.001, 0.001]
    stars:    ['n.s.', '*', '**', '***', '****']
    
    Inputs
    ------
    pvals : float or list/np.ndarray of floats
    
    Returns
    -------
    stars : string or list of strings
    '''
    thresholds = [np.inf, 0.05, 0.01, 0.001, 0.001, 0.0001]
    stars_strings = ['n.s.', '*', '**', '***', '****']
    float_flag = False
    
    stars = []
        
    if not isinstance(pvals, list) and not isinstance(pvals, np.ndarray):
        pvals = [pvals]
        float_flag = True
    for i, pval in enumerate(pvals):
        if pval < 0:
            raise ValueError(f'pvalues must not be negative but found {pval}')
        j = -1
        thresh = thresholds[0]
        while (j < len(stars_strings)-1) and pval < thresh:
            j += 1
            thresh = thresholds[j+1]
        stars.append(stars_strings[j])
    
    if float_flag:
        return stars[0]
    return stars

