from copy import deepcopy
import numpy as np
from scipy.stats import ttest_ind
from mne.stats import fdr_correction

from naplib.utils import _parse_outstruct_args
from naplib.data import Data
from naplib import logger


def responsive_ttest(data=None, resp='resp', befaft='befaft', sfreq='dataf', pre_post=[1, 1], alpha=0.05, average=False, fdr_method='indep', alternative='two-sided', equal_var=True):
    '''
    Identify responsive electrodes by performing a t-test between response
    values during silence (before stimulus) compared to during speech/sound
    (after stimulus onset) [1]_.
    `scipy's ttest_ind <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_
    is used to perform the t-test. Please see their documentation for more details.

    Parameters
    ----------
    data : naplib.Data instance, optional
        Data object containing data to be normalized in one of the field.
        If not given, must give the X and y data directly as the ``X``
        and ``y`` arguments. 
    resp : str | list of np.ndarrays or a multidimensional np.ndarray
        Electrode responses to each trial, containing a portion at the
        beginning which is the response to silence before the start
        of the stimulus. Once arranged, each trial should be of
        shape (time, num_channels).
        If a string, it must specify one of the fields of the Data
        provided in the first argument. If a multidimensional array,
        first dimension indicates the trial/epochs.
    befaft : str | list of np.ndarrays or a single np.ndarray, default='befaft'
        If a string, specifies a field of the Data which contains the period
        before and after sound onset (in sec) for each trial. Otherwise,
        a list should contain the befaft period for each trial, and a single
        np.ndarray of length 2 specifies the befaft period for all trials. For
        example, befaft=np.array([0.5, 0.5]) indicates that for each trial,
        the first half second of the responses come before the stimulus onset.
        If no Data is provided, this cannot be a string.
        Note: if this is a list it must be of same length
        as the resp, so to specify the same befaft for all trials, use a np.ndarray
        of length 2.
    sfreq : str | int, default='dataf'
        The sampling frequency of the responses. If a string, specifies field of
        the Data containing the sampling frequency.
    pre_post : np.ndarray | list, default=[1, 1]
        List or array of length 2 or 4, giving the time windows (in seconds) to compare
        before and after stimulus onset. If length 2 (such as [x,y]), both numbers must be positive floats, and
        the first number specifies the window [-x, 0) relative to sound onset while the second number
        specifies the window [0, y) relative to sound onset. The responses in these two
        windows will be compared by the t-test. If length 4, the first two floats specify the start
        and end points of the first window (relative to sound onset), and the remaining two floats
        specify the start and end points of the second window. Thus, the following two inputs
        produce the same windows and the same test, [0.8, 1] and [-0.8, 0, 0, 1], whereby the responses in the
        0.8 seconds immediately preceeding stimulus onset are compared to the responses in
        the 1 second immediately following stimulus onset. Sometimes, sound does not onset
        until a little later after stimulus onset, and there is also neural delay for many electrodes,
        so one good `pre_post` option could be [-1, 0, 0.2, 1.2] to add a buffer of 200ms for the
        electrodes to begin responding more strongly.
    alpha : float, default=0.05
        Error rate.
    average : bool, default=False
        Whether to average each segment before t-test or not. Should only be used if have a sufficiently
        high number of trials, because this will create one comparison per trial.
    fdr_method : str, {'indep', 'negcorr', None}, default='indep'
        Method of correction for multiple comparisons. If 'indep' it implements 
        Benjamini/Hochberg for independent or if 'negcorr' it corresponds
        to Benjamini/Yekutieli. If None, no false discovery rate correction is performed.
    alternative : str, {‘two-sided’, ‘less’, ‘greater’}, default='two-sided'
        The  the alternative hypothesis. Must be one of the following:
        * 'two-sided': the means of the distributions of the response values before
          and during sound are unequal.
        * 'less': the mean of the distribution of response values before stimulus onset
          is less than the mean of the distribution of response values after stimulus onset
        * 'greater': the mean of the distribution of response values before stimulus onset
          is greater than the mean of the distribution of response values after stimulus onset
    equal_var : bool, default=True
        If True, perform a standard independent 2 sample test
        that assumes equal population variances [2]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [3]_.

    Returns
    -------
    out : list of np.arrays, same as `resp` input type
        A list of numpy arrays, each of shape (time, new_num_channels)
    stats : dict
        Dictionary containing statistics about responsive elecs, with the
        following keys
        - 'pval' : p-values, shape (num_channels,)
        - 'stat' : test statistic, shape (num_channels,)
        - 'significant': True if null hypothesis was rejected (response is significantly different), False if not, shape (num_channels,)
        - 'alpha': error rate of the test

    Examples
    --------
    >>> import naplib as nl
    >>> data = nl.io.load_speech_task_data()
    >>> # Get responsiveness of the 10 electrodes, assuming that they must show an increase
    >>> # response to the stimulus, and averaging segment windows
    >>> new_data, stats = responsive_ttest(data, average=True, alternative='less')
    >>> # All 10 electrodes have significant responses
    >>> stats['significant']
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True])


    References
    ----------
    .. [1] Mesgarani, N., & Chang, E. F. (2012). Selective cortical representation
           of attended speaker in multi-talker speech perception. Nature, 485(7397), 233-236.
    .. [2] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
    .. [3] https://en.wikipedia.org/wiki/Welch%27s_t-test
    '''

    if not isinstance(pre_post, (np.ndarray, list)) or len(pre_post) not in [2, 4]:
        raise ValueError('pre_post must be an array or list of length 2 or 4')

    if len(pre_post) == 2:
        if pre_post[0] <= 0 or pre_post[1] <= 0:
            raise ValueError(f'pre_post must be all positive if length 2 but got {pre_post}')
        pre_post = [-pre_post[0], 0, 0, pre_post[1]]

    if fdr_method is not None and fdr_method not in ['indep', 'negcorr']:
        raise ValueError(f"fdr_method should be 'indep' or 'negcorr' but got {fdr_method}")

    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f"alternative must be one of ['two-sided', 'less', 'greater'] but got {alternative}")

    resp, befaft, sfreq = _parse_outstruct_args(data, resp, befaft, sfreq,
                                         allow_different_lengths=True,
                                         allow_strings_without_outstruct=False)

    if isinstance(resp, np.ndarray):
        resp = [r for r in resp]

    if not isinstance(resp, list):
        raise TypeError(f'resp parameter must specify a list of trials or a multidimensional array, but got {type(resp)}')

    if not all([r.ndim == 2 for r in resp]):
        raise ValueError(f'Some response trials are not 2-dimensional (time by channels)')

    # For each trial, pick out the regions before and immediately after stimulus onset
    before_samples = []
    after_samples = []
    pvals = []
    statistics = []
    for t in range(len(resp)):
        bef = round(befaft[t][0] * sfreq[t])

        pre_start = round(pre_post[0] * sfreq[t]) + bef
        pre_end = round(pre_post[1] * sfreq[t]) + bef
        post_start = round(pre_post[2] * sfreq[t]) + bef
        post_end = round(pre_post[3] * sfreq[t]) + bef
        
        if pre_start < 0:
            logger.warning("pre_post too large for the data's befaft period. Changing pre_post[0] so that it begins as early as possible for this data. This will result in a different size window for pre-stimulus compared to post-stimulus. Try using a smaller pre_post.")
            pre_start = 0

        if not (pre_start < pre_end and pre_end <= post_start and post_start < post_end):
            raise ValueError('pre_post provided resulted in a non-increasing time window.') 

        if bef < 5:
            raise ValueError(f'befaft period is too short, there must be at least 5 samples of response before stimulus onset.')
        before_tmp = resp[t][pre_start:pre_end] - resp[t].mean(0, keepdims=True)
        after_tmp = resp[t][post_start:post_end] - resp[t].mean(0, keepdims=True)
        if average:
            before_samples.append(before_tmp.mean(0, keepdims=True))
            after_samples.append(after_tmp.mean(0, keepdims=True))
        else:
            before_samples.append(before_tmp)
            after_samples.append(after_tmp)

    before_samples_cat = np.concatenate(before_samples, axis=0)
    after_samples_cat = np.concatenate(after_samples, axis=0)
    
    statistics, pvals = ttest_ind(before_samples_cat, after_samples_cat,
                                  equal_var=equal_var, alternative=alternative)

    if fdr_method is not None:
        reject, pval_corrected = fdr_correction(pvals, alpha=alpha, method=fdr_method)
    else:
        reject = pvals < alpha
        pval_corrected = pvals

    resp_corrected = [r[:,reject] for r in resp]

    stats = {'pval': pval_corrected, 'stat': statistics, 'significant': reject, 'alpha': alpha}

    return resp_corrected, stats
