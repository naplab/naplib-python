import pickle
import warnings
import numpy as np
from tqdm import tqdm
from hdf5storage import loadmat, savemat
import h5py

from ..data import Data

ACCEPTED_CROP_BY = ['onset', 'durations']

def import_data(filepath, strict=True, useloadmat=True, verbose=False):
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
    verbose : boolean, default=False
        If True, print trial number and field as it's loaded

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
                if verbose: print(tt, f)
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
                if verbose: print(trial, fld)
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
        warnings.warn(f'The filepath does not end with ".mat". Saving anyway. However, the .mat extension may be needed to open the file in MATLAB.')
    
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

    '''
    
    if not filename.endswith('.pkl') and '.' not in filename:
        filename = filename + '.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_bids(root,
              subject,
              datatype,
              task,
              suffix,
              session=None,
              befaft=[0, 0],
              crop_by='onset',
              info_include=['sfreq', 'ch_names'],
              resp_channels=None):
    '''
    Read data from the `BIDS file structure <https://bids.neuroimaging.io/>`_ [1]
    to create a Data object. The BIDS file structure is a commonly used structure
    for storing neural recordings such as EEG, MEG, or iEEG.
    
    The channels in the BIDS files are either stored in the 'resp' field of the
    Data object or the 'stim' field, depending on whether the `channel_type` is 'stim'.
    
    Please see the :ref:`Importing Data <import data examples>` for more detailed
    tutorials which show how to import external data.
    
    Parameters
    ----------
    root : string, path-like
        Root directory of BIDS file structure.
    datatype : string
        Likely one of ['meg','eeg','ieeg'].
    task : string
        Task name.
    suffix : string
        Suffix name in file naming. This is often the same as datatype.
    session : string
        Session name.
    befaft : list or array-like or length 2, default=[0, 0]
        Amount of time (in sec.) before and after each trial's true duration to include
        in the trial for the Data. For example, if befaft=[1,1] then if each trial's
        recording is 10 seconds long, each trial in the resulting Data object will contain
        12 seconds of data, since 1 second of recording before the onset of the event
        and 1 second of data after the end of the event are included on either end.
    crop_by : string, default='onset'
        One of ['onset', 'durations']. If crop by 'onset', each trial is split
        by the onset of each event defined in the BIDS file structure and each
        trial ends when the next trial begins. If crop by 'durations', each trial is split
        by the onset of each event defined in the BIDS file structure and each
        trial lasts the duration specified by the event. This is typically not desired
        when the events are momentary stimulus presentations that have very short duration
        because only the responses during the short duration of the event will be saved, and
        all of the following responses are truncated.
    info_include : list of strings, default=['sfreq, ch_names']
        List of metadata info to include from the raw info. For example, you may wish to include
        other items such as 'file_id', 'line_freq', etc, for later use, if they are stored in
        the BIDS data.
    resp_channels : list, default=None
        List of channel names to select as response channels to be put in the 'resp' field of
        the Data object. By default, all channels which are not of type 'stim' will be included.
        Note, the order of these channels may not be conserved.
    
    Returns
    -------
    out : Data
        Event/trial responses, stim, and other basic data in naplib.Data format.
        
    Notes
    -----
    The measurement information that is read-in by this function is stored in the Data.mne_info
    attribute. This info can be used in conjunction with
    `mne's visualization functions <https://mne.tools/stable/visualization.html>`_. 
    
    References
    ----------
    .. [1] Pernet, Cyril R., et al. "EEG-BIDS, an extension to the brain imaging
        data structure for electroencephalography." Scientific data 6.1 (2019): 1-5.
    '''
    
    try:
        from mne_bids import BIDSPath, read_raw_bids
    except Exception as e:
        raise Exception('Missing package MNE-BIDS which is required for reading data from BIDS. Please '
            'install it with "pip install --user -U mne-bids" or by following the instructions '
            'at https://mne.tools/mne-bids/stable/install.html')
    
    if crop_by not in ACCEPTED_CROP_BY:
        raise ValueError(f'Invalid "crop_by" input. Expected one of {ACCEPTED_CROP_BY} but got "{crop_by}"')
    
    bids_path = BIDSPath(subject=subject, root=root, session=session, task=task,
                         suffix=suffix, datatype=datatype)
    
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
            
    raws = _crop_raw_bids(raw, crop_by, befaft)
    
    raw_info = None
    
    # figure out which channels are stimulus channels
    stim_channels = [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type == 'stim']

    # for each trial, separate into stim and response channels
    raw_responses = []
    raw_stims = []
    for raw_trial in raws:
        raw_resp = raw_trial.copy().drop_channels(stim_channels)
        if resp_channels is not None:
            raw_resp = raw_resp.pick_channels(resp_channels, verbose=0)
        raw_responses.append(raw_resp)
        
        if raw_info is None:
            raw_info = raw_resp.info

        # if any of the channels are 'stim' channels, store them separately from responses
        if 'stim' in raw_trial.get_channel_types():
            raw_stims.append(raw_trial.pick_types(stim=True, verbose=0))
        else:
            raw_stims.append(None)
    
    # build Data
    new_data = []
    for trial in tqdm(range(len(raws))):
        trial_data = {}
        trial_data['event_index'] = trial
        if 'description' in raw_responses[trial].annotations[0]:
            trial_data['description'] = raw_responses[trial].annotations[0]['description']
        if raw_stims[trial] is not None:
            trial_data['stim'] = raw_stims[trial].get_data().transpose(1,0) # time by channels
            trial_data['stim_ch_names'] = raw_stims[trial].info['ch_names']
        trial_data['resp'] = raw_responses[trial].get_data().transpose(1,0) # time by channels
        trial_data['befaft'] = befaft
        for info_key in info_include:
            if info_key not in info_include:
                warnings.warn(f'info_include key "{info_key}" not found in raw info')
            else:
                trial_data[info_key] = raw_responses[trial].info[info_key]
        new_data.append(trial_data)  

    data_ = Data(new_data, strict=False)
    data_.set_mne_info(raw_info)
    return data_
    
    
def _crop_raw_bids(raw_instance, crop_by, befaft):
    '''
    Crop the raw data to trials based on events in its annotations.
    
    Parameters
    ----------  
    raw_instance : mne.io.Raw-like object
    
    crop_by : string, default='onset'
        One of ['onset', 'annotations']. If crop by 'onset', each trial is split
        by the onset of each event defined in the BIDS file structure and each
        trial ends when the next trial begins. If crop by 'annotations', each trial is split
        by the onset of each event defined in the BIDS file structure and each
        trial lasts the duration specified by the event. This is typically not desired
        when the events are momentary stimulus presentations that have very short duration
        because only the responses during the short duration of the event will be saved, and
        all of the following responses are truncated.
    
     Returns
     -------
     raws : list
         The cropped raw objects.

    '''

    max_time = (raw_instance.n_times - 1) / raw_instance.info['sfreq']
    
    raws = []
    for i, annot in enumerate(raw_instance.annotations):
        onset = annot["onset"] - raw_instance.first_time - befaft[0]
        if -raw_instance.info['sfreq'] / 2 < onset < 0:
            onset = 0
        if crop_by == 'onset':
            if i == len(raw_instance.annotations)-1:
                tmax = max_time
            else:
                if befaft[1] > 0:
                    warnings.warn('befaft[1] is positive, but crop_by is "onset", so the ending of each trial will include a portion of the next trial')
                tmax = raw_instance.annotations[i+1]["onset"] + befaft[1]
            tmax = min([tmax, max_time])
            raw_crop = raw_instance.copy().crop(onset, tmax)
        
        else:
            tmax = onset + annot["duration"] + befaft[1]
            tmax = min([tmax, max_time])
            raw_crop = raw_instance.copy().crop(onset, tmax)
        
        raws.append(raw_crop)
    
    return raws
