import numpy as np
from hdf5storage import loadmat
from ..out_struct import OutStruct

def import_outstruct(filepath, strict=True):
    '''
    Import outstruct from matlab (.mat) format. Transpose 'resp' field
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

def load(filename):
    '''
    Load OutStruct or other object from saved file.
    
    Parameters
    ----------
    filename : string
        File to load. If doesn't end with .pkl this will be added
        automatically.
    
    Returns
    -------
    output : naplib.OutStruct or other object
        Loaded object.
    
    Raises
    ------
    FileNotFoundError
        Can't find file.
    
    '''
    
    if not filename.endswith('.pkl') and '.' not in filename:
        filename = filename + '.pkl'
        
    with open(filename, 'rb') as inp:
        output = pickle.load(inp)

    return output


def save(filename, obj):
    '''
    Save OutStruct or other object with pickle.
    
    Parameters
    ----------
    filename : string
        File to load. If doesn't end with .pkl this will be added
        automatically.
    obj : OutStruct or other object
        Data to save.
    
    Returns
    -------
    
    '''
    
    if not filename.endswith('.pkl') and '.' not in filename:
        filename = filename + '.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(out, f)
