from copy import deepcopy
import numpy as np

from ..out_struct import OutStruct

def normalize(data, field=None, axis=-1, method='zscore'):
    '''Normalize data.
    
    Parameters
    ----------
    data : either a naplib.OutStruct object, list of np.ndarrays, or multidimensional np.ndarray
        Data to normalize. If a multidimensional array, first dimension indicates the trial/instances,
        which will be concatenated over to compute normalization statistics.
        
    field : string, default=None (optional)
        If data is a naplib.OutStruct object, this must be provided. Determines which field of the OutStruct
        is normalized.
    
    axis : int, default=-1
        axis of the array to normalize over.
    
    method : string, default='zscore'
        Method of normalization. Must be one of ['zscore','center'].
        'center' only centers the data, while 'zscore' also scales by standard deviation
    
    Returns
    -------
    normalized_data : object of same type as input data
    
    '''
    
    data2 = deepcopy(data)
    
    if method not in ['zscore', 'center']:
        raise ValueError(f"Bad method input. method must be one of ['zscore', 'center'], but found {method}")

    if isinstance(data, OutStruct):
        field_data = data.get_field(field)
    elif isinstance(data, np.ndarray):
        field_data = [d for d in data]
        axis -= 1 # since we got rid of the first axis by putting into a list, need to change this
    elif isinstance(data, list):
        field_data = data
    else:
        raise TypeError(f'data input must be of type OutStruct, np.ndarray, or list or arrays, but found {type(data)}')

    concat_data = np.concatenate(field_data, axis=axis)
    eps = 1e-13
    
    center_val = np.mean(concat_data, axis=axis, keepdims=True)
    if method=='zscore':
        std_val = np.std(concat_data, axis=axis, keepdims=True)
    
    for i, tmp in enumerate(field_data):
        
        if isinstance(data, OutStruct):
            if method=='zscore':
                data2[i][field] = (tmp - center_val) / std_val
            elif method=='center':
                data2[i][field] = tmp - center_val
        else:
            if method=='zscore':
                data2[i] = (tmp - center_val) / std_val
            elif method=='center':
                data2[i] = tmp - center_val
   
    return data2
    

