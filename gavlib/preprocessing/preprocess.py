from copy import deepcopy
import numpy as np

from ..out_struct import OutStruct
from ..utils import _parse_outstruct_args

def normalize(outstruct=None, data='resp', axis=0, method='zscore'):
    '''Normalize data over multiple trials. If you are trying to normalize a single matrix,
    it must be passed in inside a list, or with an extra first dimension.
    
    Parameters
    ----------
    outstruct : naplib.OutStruct object, optional
        OutStruct containing data to be normalized in one of the field. If not given, must give
        the data to be normalized directly as the ``data`` argument. 

    data : string if outstruct given, or a list of np.ndarrays or a multidimensional np.ndarray
        Data to normalize. If a string, it must specify one of the fields of the outstruct
        provided in the first argument. If a multidimensional array, first dimension
        indicates the trial/instances which will be concatenated over to compute
        normalization statistics.
    
    axis : int, default=-1
        Axis of the array to normalize over.
    
    method : string, default='zscore'
        Method of normalization. Must be one of ['zscore','center'].
        'center' only centers the data, while 'zscore' also scales by standard deviation
    
    Returns
    -------
    normalized_data : list of np.ndarrays
    
    '''
    
    data = _parse_outstruct_args(outstruct, data)
    
    if method not in ['zscore', 'center']:
        raise ValueError(f"Bad method input. method must be one of ['zscore', 'center'], but found {method}")

    if isinstance(data, np.ndarray):
        data = [d for d in data]
        axis -= 1 # since we got rid of the first axis by putting into a list, need to change this
    elif not isinstance(data, list) or not isinstance(data[0], np.ndarray):
        raise TypeError(f'data found is not either np.ndarray, or list or arrays, but found {type(data)}')

    concat_data = np.concatenate(data, axis=axis)
    
    center_val = np.mean(concat_data, axis=axis, keepdims=True)
    if method=='zscore':
        std_val = np.std(concat_data, axis=axis, keepdims=True)
    
    for i, tmp in enumerate(data):
        if method=='zscore':
            data[i] = (tmp - center_val) / std_val
        elif method=='center':
            data[i] = tmp - center_val

    return data

