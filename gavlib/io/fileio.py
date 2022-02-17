import numpy as np
from hdf5storage import loadmat
from ..out_struct import OutStruct

def import_out_struct(filepath, strict=True):
    '''
    Import out structure from matlab format. Transpose 'resp' field
    so that it is shape (time, channels)

    Parameters
    ----------
    filepath : string
        Path to .mat file with out structure.
    strict : bool, default=True
        If True, requires the data to at least contain all the following
        fields: ['name','sound','soundf','resp','dataf']
    Returns
    -------
    out : naplib.OutStruct object
    '''
    loaded = loadmat(filepath)
    loaded = loaded['out'].squeeze()
    fieldnames = loaded[0].dtype.names
    
    req = ['name','sound','soundf','resp','dataf']
    
    
    data = []
    for trial in loaded:
        trial_dict = {}
        for f, t in zip(fieldnames, trial):
            tmp_t = t.squeeze()
            if f == 'resp':
                tmp_t = tmp_t.transpose()
            try:
                tmp_t = tmp_t.item()
            except:
                pass
            trial_dict[f] = tmp_t
        data.append(trial_dict)
    
    fieldnames = set(data[0].keys())
    for r in req:
        if strict and r not in fieldnames:
            raise ValueError(f'Missing required field: {r}')
    
    out = OutStruct(data=data)
    return out
