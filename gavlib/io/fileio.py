import numpy as np
from hdf5storage import loadmat
from ..out_struct import OutStruct

def import_out_struct(filepath):
    '''
    Import out structure from matlab format.

    Parameters
    ----------
    filepath : string

    Returns
    -------
    out : naplib.OutStruct object
    '''
    loaded = loadmat(filepath)
    loaded = loaded['out'].squeeze()
    fieldnames = loaded[0].dtype.names
    
    data = []
    for trial in loaded:
        trial_dict = {}
        for f, t in zip(fieldnames, trial):
            tmp_t = t.squeeze()
            if tmp_t.dtype == '<U6':
                tmp_t = str(tmp_t)
            trial_dict[f] = tmp_t
        data.append(trial_dict)
    
    out = OutStruct(data=data)
    return out
