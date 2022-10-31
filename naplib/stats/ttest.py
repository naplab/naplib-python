from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats import t as scipyT
import statsmodels.api as sm
from patsy import dmatrices, dmatrix


def ttest(*args, classes=None, cat_feats={}, con_feats={}, return_ols_result=False):
    '''
    Perform a t-test under the linear model framework, allowing for additional features
    to be controlled. For example, performing a paired t-test between samples
    while controlling for the group identity that each sample comes from.

    If no categorical or continuous features are given, then this function can reproduce
    scipy's `ttest_1samp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html>`_, 
    `ttest_rel <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html>`_, 
    or `ttest_ind <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`_.
    Please see the Examples below for details on this equivalence.  However, if any
    additional features are provided, they are treated as extra predictors in a linear model
    framework, therefore controlling for their impact when estimating the test statistic.

    In the case of neural data, a common use-case for this is when testing a distribution of data points,
    each of which comes from a group, like a subject. For example, we may have 3 subjects, each of which
    has 10 electrodes, and we want to perform a relative t-test between each electrode's value
    during condition A and condition B. This function allows us to perform that test while controlling
    for the impact of subject identity by providing a subject identity categorical feature to the model
    which it uses to aid it its prediction.
 
    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Data to test. All samples may come from a single class, in which case a one sample t-test
        is performed compared to a null hypothesis of mean=0, or from multiple classes if ``classes``
        is provided as well, in which case an independent two-sample t-test is performed.
    y : array-like, shape (n_samples,), optional
        If provided, must be the same shape as *x* and a paired t-test is performed between
        *x* and *y*.
    classes : array-like, shape (n_samples,), optional
        An array of 0's and 1's to separate the two classes to compare in an
        independent two-sample t-test if only *x* is provided without *y*.
    cat_feats : array-like, dict or DataFrame, optional
        Categorical features to control for, given as an array-like or dict-like object (including a
        dict, a pd.DataFrame, etc). If given as dict-like, each key specifies the name of the
        factor to control for, and the value is an array-like of shape (n_samples,) of that factor's values.
        If an array, then each column provides a categorical factor to control for.
        All features are automatically one-hot encoded. If performing an independent 2-sample
        ttest where `classes` were provided, then if this is dict-like, it cannot contain a key called "classes".
    con_feats : array-like, dict or DataFrame, optional
        Continuous features to control for, given as an array-like or dict-like object (including a
        dict, a pd.DataFrame, etc). If given as dict-like, each key specifies the name of the
        factor to control for, and the value is an array-like of shape (n_samples,) of that factor's values.
        If an array, then each column provides a continuous factor to control for. These features
        are converted to floats automatically.
        Key names must not overlap with cat_feats if given as dict-like.
    return_ols_result : bool, default=False
        Whether to return the ols result object as a third item in the tuple.
    
    Returns
    -------
    statistic : float
        T-value statistic.
    pvalue : float
        p-value.
    ols_result : statsmodels.regression.linear_model.RegressionResultsWrapper
        Only returned if ``return_ols_result=True``
    
    Examples
    --------
    >>> from naplib.stats import ttest
    >>> import numpy as np
    >>> rng = np.random.default_rng(10)
    >>> # Imagine we have data that comes from two underlying subjects pooled together
    >>> x1 = rng.normal(size=(10,1))    # subject 1 data
    >>> x2 = rng.normal(size=(10,1))+4  # subject 2 data
    >>> x = np.concatenate([x1,x2])     # pooled data
    >>> x1_group = np.zeros((10,))
    >>> x2_group = np.ones((10,))
    >>> groups = np.concatenate([x1_group,x2_group])    # pooled subject identity vector
    >>> # Also, each data point from each subject comes from one of three behavior types
    >>> behavior = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])

    We can perform a 1-sample t-test on all the x values assuming a null
    population mean of 0. This is equivalent to
    scipy.stats.ttest_1samp(x, popmean=0)

    >>> ttest(x)
    (3.9115219870652487, 0.0009378023836864556)
    
    Or we can do the same test but take into account the subject identity
    of the samples.
    
    >>> ttest(x, cat_feats=groups)
    (-0.33532229593097984, 0.7412584236348754)
    
    If x1 and x2 are paired data, we can do a paired relative t-test.
    This is equivalent to scipy.stats.ttest_rel(x2, x1)
    
    >>> ttest(x2, x1)
    (11.052116218441915, 1.5470048033033527e-06)

    Or the same test but take into account the behavior type of each sample.
    
    >>> ttest(x2, x1, cat_feats=behavior)
    (11.761977075053782, 7.272560323998328e-06)
    
    If x1 and x2 are not paired, we can do an independent 2 sample t-test.
    This is equivalent to scipy.stats.ttest_ind(x2, x1)
    
    >>> ttest(x, classes=groups)
    (11.335049583473713, 1.256342249131581e-09)

    Or the same test but take behavior into account. Since the cat_feats parameter
    must have the same number of samples as the input data, we need to concatenate
    the behavior so that we have a behavior indicator for each sample in x.

    >>> behavior_x1x2 = np.concatenate([behavior, behavior], axis=0)
    >>> ttest(x, classes=groups, cat_feats=behavior_x1x2)
    (11.682487719722063, 3.0312825227658624e-09)
    
    '''

    if not isinstance(cat_feats, np.ndarray) and not isinstance(cat_feats, dict) and not isinstance(cat_feats, pd.DataFrame):
        raise TypeError(f'cat_feats must be a np.ndarray, dict, or DataFrame, but got {type(cat_feats)}')
    if not isinstance(con_feats, np.ndarray) and not isinstance(con_feats, dict) and not isinstance(con_feats, pd.DataFrame):
        raise TypeError(f'con_feats must be a np.ndarray, dict, or DataFrame, but got {type(con_feats)}')

    cat_feats_ = deepcopy(cat_feats)
    con_feats_ = deepcopy(con_feats)

    if isinstance(cat_feats_, pd.DataFrame):
        len_cat = len(cat_feats_.columns)
    elif isinstance(cat_feats_, np.ndarray):
        if cat_feats_.ndim == 1:
            len_cat = 1
            cat_feats_ = cat_feats_[:,np.newaxis]
        cat_feats_ = dict(zip([f'cat_{i}' for i in range(len_cat)], cat_feats_.T))
    else: # already a dict
        len_cat = len(cat_feats_)
    
    
    if isinstance(con_feats_, pd.DataFrame):
        len_con = len(con_feats_.columns)
    elif isinstance(con_feats_, np.ndarray):
        if con_feats_.ndim == 1:
            con_feats_ = con_feats_[:,np.newaxis]
        len_con = con_feats_.shape[1]
        con_feats_ = dict(zip([f'con_{i}' for i in range(len_con)], con_feats_.T))
    else: # already a dict
        len_con = len(con_feats_)
        
    test_type = None
        
    if len(args) == 1:
        y = np.asarray(args[0])
        if classes is not None:
            # independent 2-sample ttest
            assert 'classes' not in cat_feats_, 'the key name "classes" cannot be in the control dict'
            if not np.array_equal(sorted(np.unique(np.asarray(classes))), np.array([0,1])):
                raise ValueError(f'classes must be an array of only zeros and ones')
            cat_feats_['classes'] = np.asarray(classes).astype('int')
            test_type = "ind"
        else:
            test_type = "1_samp"
    elif len(args) == 2:
        x, y = np.asarray(args[0]), np.asarray(args[1])
        assert len(x) == len(y), 'x and y must be the same length'
        y = x - y
        test_type = 'rel'
    else:
        raise ValueError(f'Must provide either 1 or 2 positional arguments (x and y), but got {len(args)}')
    
    new_data = {}
    for k in cat_feats_.keys():
        new_data[k] = cat_feats_[k].astype('str')
    for k in con_feats_.keys():
        new_data[k] = con_feats_[k].astype('float')
        
    if len(new_data) < len_con + len_cat:
        raise ValueError(f'cat_feats and con_feats have overlapping key names.')

        
    formula = ""
    for k in new_data.keys():
        formula += k + " + "
    formula = formula [:-2] # remove the last +

    
    if (test_type == 'rel' or test_type == '1_samp') and len(new_data) == 0:
        X = np.ones((len(y),1)) # need a dummy for the intercept
        names = ['Intercept']
    else:
        X = dmatrix(formula, new_data)
        names = X.design_info.column_names

#     print(X)
    
    X_df = pd.DataFrame(dict(zip(names, X.astype('float').T)))
    
    ols = sm.OLS(y, X_df)
    ols_result = ols.fit()
    
#     print(ols_result.summary())
    
    if test_type == 'rel' or test_type == '1_samp':
        tval = ols_result.tvalues['Intercept']
        pval = ols_result.pvalues['Intercept']
    else:
        tval = ols_result.tvalues['classes[T.1]']
        pval = ols_result.pvalues['classes[T.1]']
    
    if return_ols_result:
        return tval, pval, ols_result
    else:
        return tval, pval

