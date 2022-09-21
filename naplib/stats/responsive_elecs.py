from copy import deepcopy
import numpy as np
from scipy.stats import ttest_ind
from mne.stats import fdr_correction

from ..utils import _parse_outstruct_args
from ..data import Data


def responsive_ttest(data=None, resp='resp', befaft='befaft', sfreq='dataf', alpha=0.05, fdr_method='indep', alternative='two-sided', equal_var=True, random_state=None):
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
        If a string, specifies a field of the Data which contains
        the before and after time (in sec) for each trial. Otherwise,
        a list should contain the befaft period for each trial, and a single
        np.ndarray of length 2 specifies the befaft period for all trials. For
        example, befaft=np.array([0.5, 0.5]) indicates that for each trial,
        the first half second of the `resp` is the responses before the onset
        of the stimulus, and also the final half second is responses for half
        a second after the stimulus ended. If no Data is provided, this
        cannot be a string. Note: if this is a list it must be of same length
        as the resp, so to specify the same befaft for all trials, use a np.ndarray
        of length 2.
    sfreq : str | int, default='dataf'
        The sampling frequency of the responses. If a string, specifies field of
        the Data containing the sampling frequency.
    alpha : float, default=0.05
        Error rate.
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
    random_state : int, default=None
        Random seed which can be set for reproducibility.

    Returns
    -------
    out : naplib.Data | list of np.arrays, same as `resp` input type
        If Data object was given as input, this is a copy of that
        Data with the `resp` field replaced by only the responsive
        electrode data. If `resp` input was given as a list of arrays
        or a 3D array, then this is a list of numpy arrays, each of shape
        (time, new_num_channels)
    stats : dict
        Dictionary containing statistics about responsive elecs, with the
        following keys
        - 'pval' : p-values, shape (num_channels,)
        - 'stat' : test statistic, shape (num_channels,)
        - 'significant': True if null hypothesis was rejected (response is significantly different), False if not, shape (num_channels,)
        - 'alpha': error rate of the test


    References
    ----------
    .. [1] Mesgarani, N., & Chang, E. F. (2012). Selective cortical representation
           of attended speaker in multi-talker speech perception. Nature, 485(7397), 233-236.
    .. [2] https://en.wikipedia.org/wiki/T-test#Independent_two-sample_t-test
    .. [3] https://en.wikipedia.org/wiki/Welch%27s_t-test

    '''

    if fdr_method is not None and fdr_method not in ['indep', 'negcorr']:
        raise ValueError(f"fdr_method should be 'indep' or 'negcorr' but got {fdr_method}")

    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f"alternative must be one of ['two-sided', 'less', 'greater'] but got {alternative}")

    if isinstance(data, Data):
        return_as_data = True
        data_copy = deepcopy(data)
        if isinstance(resp, str):
            resp_fieldname = deepcopy(resp) # if this is a string, need to save the name now
        else:
            resp_fieldname = 'resp'
    else:
        return_as_data = False

    resp, befaft, sfreq = _parse_outstruct_args(data, deepcopy(resp), befaft, sfreq,
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
    rng = np.random.default_rng(random_state)
    for t in range(len(resp)):
        bef = int(befaft[t][0] * sfreq[t])
        if bef < 5:
            raise ValueError(f'befaft period is too short, there must be at least 3 samples of response before stimulus onset.')
        before_samples.append(resp[t][:bef])
        after_samples.append(resp[t][bef:3*bef])

    before_samples = np.concatenate(before_samples, axis=0)
    after_samples = np.concatenate(after_samples, axis=0)

    N_retest = 20
    # do the test N_retest times and average the stats
    for _ in range(N_retest):
        before_samples_permuted = rng.permuted(before_samples, axis=0)
        after_samples_permuted = rng.permuted(after_samples, axis=0)
        N_test_samples = min([before_samples_permuted.shape[0]//2, after_samples_permuted.shape[0]//2])
        stat, pval = ttest_ind(before_samples_permuted[:N_test_samples], after_samples_permuted[:N_test_samples], axis=0,
                               equal_var=equal_var, alternative=alternative)
        pvals.append(pval)
        statistics.append(stat)

    statistics = np.array(statistics).mean(0)
    pvals = np.array(pvals).mean(0)

    if fdr_method is not None:
        reject, pval_corrected = fdr_correction(pvals, alpha=alpha, method=fdr_method)
    else:
        reject = pvals < alpha
        pval_corrected = pvals

    resp_corrected = [r[:,reject] for r in resp]

    stats = {'pval': pval, 'stat': statistics, 'significant': reject, 'alpha': alpha}

    if return_as_data:
        data_copy[resp_fieldname] = resp_corrected
        return data_copy, stats

    return resp_corrected, stats


    








