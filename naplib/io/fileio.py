import pickle
import numpy as np
from hdf5storage import loadmat, savemat
import h5py

from naplib import logger
from ..data import Data

def import_data(filepath, strict=True, useloadmat=True):
    '''
    Import Data object from MATLAB (.mat) format. This will
    automatically transpose the 'resp' and 'aud' fields
    so that they are shape (time, channels) for each trial. The
    MATLAB equivalent structure is a 1xN struct with N trials and
    some number of fields, and this is stored in the .mat file
    under the variable name "out".

    Parameters
    ----------
    filepath : string
        Path to .mat file.
    strict : bool, default=True
        If True, requires strict adherance to the following standards:
        1) Each trial must contain at least the following fields:
        ['name','sound','soundf','resp','dataf']
        2) Each trial must contain the exact same set of fields
    useloadmat : boolean, default=True
        If True, use hdf5storage.loadmat, else use custom h5py loader

    Returns
    -------
    data : naplib.Data object
    
    Notes
    -----
    Given the highly-specific nature of the Data object Matlab format, this
    function is mostly used internally by Neural Acoustic Processing
    Lab members.
    '''
    req = ['name','sound','soundf','resp','dataf']
    data = []
    if useloadmat:
        loaded = loadmat(filepath)
        loaded = loaded['out'].squeeze()
        fieldnames = loaded[0].dtype.names

        for tt,trial in enumerate(loaded):
            trial_dict = {}
            for f, t in zip(fieldnames, trial):
                logger.debug(f'Loading trial #{tt}: {f}')
                tmp_t = t.squeeze()
                if f == 'resp' or f == 'aud':
                    if tmp_t.ndim > 1:
                        tmp_t = tmp_t.transpose(1,0,*[i for i in range(2, tmp_t.ndim)]) # only switch the first 2 dimensions if there are more than 2
                try:
                    tmp_t = tmp_t.item()
                except:
                    pass
                trial_dict[f] = tmp_t
            data.append(trial_dict)
    else:
        f = h5py.File(filepath)
        fieldnames = list(f['out'].keys())
        n_trial = f['out'][fieldnames[0]].shape[0]
    
        for trial in range(n_trial):
            trial_dict = {}
            for fld in fieldnames:
                logger.debug(f'Loading trial #{trial}: {fld}')
                tmp = np.array(f[f['out'][fld][trial][0]])
                # Pull out scalars
                if np.prod(tmp.shape) == 1:
                    tmp = tmp[0,0]
                else:
                    try:
                        tmp = ''.join([chr(c[0]) for c in tmp])
                    except:
                        # Read cell arrays within entries
                        if isinstance(tmp[0,0], h5py.h5r.Reference):
                            shp = tmp.shape
                            tmp_flat = np.ravel(tmp)
                            for tt in range(len(tmp_flat)):
                                # Handle cell arrays containing strings
                                try:
                                    tmp_flat[tt] = ''.join([chr(c[0]) for c in f[tmp_flat[tt]][:]])
                                except:
                                    tmp_flat[tt] = f[tmp_flat[tt]][:]
                            tmp = np.reshape(tmp_flat, shp)
                            # Remove lists with single item
                            try:
                                while len(tmp) == 1:
                                    tmp = tmp[0]
                            except:
                                pass
                        tmp = np.squeeze(tmp)

                trial_dict[fld] = tmp
            data.append(trial_dict)
    
    for r in req:
        if strict and r not in fieldnames:
            raise ValueError(f'Missing required field: {r}')
    
    out = Data(data=data, strict=strict)
    return out

def export_data(filepath, data, fmt='7.3'):
    '''
    Export a naplib.Data instance to the MATLAB-compatible
    equivalent (.mat file).
    The MATLAB equivalent structure is a 1xN struct with N trials and
    some number of fields, and this is stored in the .mat file
    under the variable name "out". This function will
    automatically transpose the 'resp' and 'aud' fields for
    each trial in the .mat file, thus undoing the actions of
    import_data.

    Parameters
    ----------
    filepath : string
        Filename or path-like specifying where to save the file.
    data : Data instance
        Data to export.
    fmt : str, default='7.3'
        MATLAB file format. Options are {'7.3','7','6'}
    
    '''
    if not filepath.endswith('.mat'):
        logger.warning(f'The filepath does not end with ".mat". Saving anyway. However, the .mat extension may be needed to open the file in MATLAB.')
    
    FORMAT_OPTIONS = ['7.3','7','6']
    if fmt not in FORMAT_OPTIONS:
        raise ValueError(f"format must be one of ['7.3','7','6'] but got {fmt}")
    if not isinstance(data, Data):
        raise TypeError(f'data must be a naplib.Data instance but got {type(data)}')
    
    fieldnames = data.fields

    dt = np.dtype([(field, 'O') for field in data.fields])
    
    # construct a numpy void array which contains multiple dtypes
    void_data = []
    for trial in data:
        trial_data = []
        for field in fieldnames:
            trial_tmp = trial[field]

            expand_dimension = 0
            if isinstance(trial_tmp, np.ndarray):
                expand_dims = False if trial_tmp.ndim > 1 else True
                if trial_tmp.ndim == 1:
                    expand_dimension = 1 # column vec for matlab
                if (field == 'resp' or field == 'aud') and trial_tmp.ndim > 1:
                        trial_tmp = trial_tmp.transpose(1,0,*[i for i in range(2, trial_tmp.ndim)])
            else:
                expand_dims = True
            
                # check for other object types
                if isinstance(trial_tmp, str):
                    trial_tmp = np.array(trial_tmp, dtype='str')
                elif isinstance(trial_tmp, list):
                    trial_tmp = np.array(trial_tmp)
                    expand_dimension = 0
                elif isinstance(trial_tmp, int):
                    trial_tmp = np.array(trial_tmp, dtype='float').reshape((1,))
                else:
                    trial_tmp = np.array(trial_tmp)

            if expand_dims:
                trial_tmp = np.expand_dims(trial_tmp, expand_dimension)

            trial_data.append(trial_tmp)
        void_data.append(tuple(trial_data))
    void_data = np.array(void_data, dtype=dt).reshape(1,-1)
    
    savemat(filepath, {'out': void_data}, appendmat=False, format=fmt)


def load(filename):
    '''
    Load object from saved file.
    
    Parameters
    ----------
    filename : string
        File to load. If doesn't end with .pkl this will be added
        automatically.
    
    Returns
    -------
    output : Object
        Loaded object.
    
    Raises
    ------
    FileNotFoundError
        Can't find file.

    Examples
    --------
    >>> from naplib.io import save, load
    >>> arr = [1, 2, 3]
    >>> save('data.pkl', arr)
    >>> arr_loaded = load('data.pkl')
    >>> arr_loaded
    [1, 2, 3]
    
    '''
    
    if not filename.endswith('.pkl') and '.' not in filename:
        filename = filename + '.pkl'
        
    with open(filename, 'rb') as inp:
        output = pickle.load(inp)

    return output


def save(filename, obj):
    '''
    Save object with pickle.
    
    Parameters
    ----------
    filename : string
        File to load. If doesn't end with .pkl this will be added
        automatically.
    obj : Object
        Data to save.

    Examples
    --------
    >>> from naplib.io import save, load
    >>> arr = [1, 2, 3]
    >>> save('data.pkl', arr)
    >>> arr_loaded = load('data.pkl')
    >>> arr_loaded
    [1, 2, 3]

    '''
    
    if not filename.endswith('.pkl') and '.' not in filename:
        filename = filename + '.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

