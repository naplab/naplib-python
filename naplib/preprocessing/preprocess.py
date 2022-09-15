from copy import deepcopy
import numpy as np

from ..data import Data
from ..utils import _parse_outstruct_args

def normalize(data=None, field='resp', axis=0, method='zscore'):
    '''Normalize data over multiple trials. If you are trying to normalize a single matrix,
    it must be passed in inside a list, or with an extra first dimension.
    
    Parameters
    ----------
    data : naplib.Data object, optional
        Data object containing data to be normalized in one of the field. If not given, must give
        the data to be normalized directly as the ``data`` argument. 

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
    
    Returns
    -------
    normalized_data : list of np.ndarrays
    
    '''
    
    data_ = _parse_outstruct_args(data, field)
    
    if method not in ['zscore', 'center']:
        raise ValueError(f"Bad method input. method must be one of ['zscore', 'center'], but found {method}")

    if isinstance(data_, np.ndarray):
        data_ = [d for d in data_]
        axis -= 1 # since we got rid of the first axis by putting into a list, need to change this
    elif not isinstance(data_, list) or not isinstance(data_[0], np.ndarray):
        raise TypeError(f'data found is not either np.ndarray, or list or arrays, but found {type(data_)}')

    if axis is None:
        concat_data = np.concatenate(data_, axis=0)
        center_val = np.mean(concat_data, axis=axis, keepdims=False)
    else:
        concat_data = np.concatenate(data_, axis=axis)
        center_val = np.mean(concat_data, axis=axis, keepdims=True)
    
    if method=='zscore':
        std_val = np.std(concat_data, axis=axis, keepdims=True)
    
    for i, tmp in enumerate(data_):
        if method=='zscore':
            data_[i] = (tmp - center_val) / std_val
        elif method=='center':
            data_[i] = tmp - center_val

    return data_

