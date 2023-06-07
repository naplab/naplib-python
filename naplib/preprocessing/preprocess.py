import numpy as np
from scipy.stats import zscore

from ..array_ops import concat_apply
from ..utils import _parse_outstruct_args

def normalize(data=None, field='resp', axis=0, method='zscore', nan_policy='propagate'):
    '''Normalize data over multiple trials. If you are trying to normalize a single matrix,
    it must be passed in inside a list, or with an extra first dimension.
    
    Parameters
    ----------
    data : naplib.Data object, optional
        Data object containing data to be normalized in one of the field. If not given, then the
        the data to be normalized must be passed directly as a list of trial arrays
        to the ``field`` argument instead of a string. 

    field : string | list of np.ndarrays or a multidimensional np.ndarray
        Field to normalize. If a string, it must specify one of the fields of the Data
        provided in the first argument. If a multidimensional array, first dimension
        indicates the trial/instances which will be concatenated over to compute
        normalization statistics. If a list, all arrays will be concatenated over ``axis``.
    
    axis : int or None, default=0
        Axis of the array to concatenate over before normalizing over this same axis.
        If None, computes statistics over all dimensions jointly, thus normalizing over
        the global statistics.
    
    method : string, default='zscore'
        Method of normalization. Must be one of ['zscore','center'].
        'center' only centers the data, while 'zscore' also scales by standard deviation
    
    nan_policy : string, default='propagate'
        One of {'propagate','omit','raise'}. Defines how to handle when input
        contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.  Note that when the value is 'omit',
        nans in the input also propagate to the output, but they do not affect
        the z-scores computed for the non-nan values.
    
    Returns
    -------
    normalized_data : list of np.ndarrays
    
    '''
    
    data = _parse_outstruct_args(data, field)
    
    if method not in ['zscore', 'center']:
        raise ValueError(f"Bad method input. method must be one of ['zscore', 'center'], but found {method}")
        
    if nan_policy not in ['propagate','omit','raise']:
        raise ValueError(f"Bad nan_policy input. Must be one of ['propagate','omit','raise'], but found {nan_policy}")

    if isinstance(data, np.ndarray):
        data = [d for d in data]
        axis -= 1 # since we got rid of the first axis by putting into a list, need to change this
    elif not isinstance(data, list) or not isinstance(data[0], np.ndarray):
        raise TypeError(f'data found is not either np.ndarray, or list or arrays, but found {type(data)}')

    if axis is None:
        concat_data = np.concatenate(data, axis=0)
        if nan_policy in ['propagate', 'raise']:
            center_val = np.mean(concat_data, axis=axis, keepdims=False)
        else:
            center_val = np.nanmean(concat_data, axis=axis, keepdims=False)
    else:
        concat_data = np.concatenate(data, axis=axis)
        if nan_policy in ['propagate', 'raise']:
            center_val = np.mean(concat_data, axis=axis, keepdims=True)
        else:
            center_val = np.nanmean(concat_data, axis=axis, keepdims=True)
    del concat_data
    
    if nan_policy == 'raise':
        if np.any(np.isnan(center_val)):
            raise ValueError(f'nan found in data')
       
    if method == 'zscore':
        if axis is None:
            return concat_apply(data, zscore, axis=0, function_kwargs={'nan_policy': nan_policy, 'axis': None})
        else:
            return concat_apply(data, zscore, axis=axis, function_kwargs={'nan_policy': nan_policy, 'axis': axis})
    
    # method == 'center'
    for i, tmp in enumerate(data):
        data[i] = tmp - center_val

    return data

