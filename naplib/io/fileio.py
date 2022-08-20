import pickle
import warnings
import numpy as np
from tqdm import tqdm
from hdf5storage import loadmat

from ..out_struct import OutStruct

ACCEPTED_CROP_BY = ['onset', 'durations']

def import_outstruct(filepath, strict=True):
    '''
    Import outstruct from matlab (.mat) format. Transpose 'resp' field
    so that it is shape (time, channels).

    Parameters
    ----------
    filepath : string
        Path to .mat file with out structure.
    strict : bool, default=True
        If True, requires strict adherance to the following standards:
        1) Each trial must contain at least the following fields:
        ['name','sound','soundf','resp','dataf']
        2) Each trial must contain the exact same set of fields

    Returns
    -------
    out : naplib.OutStruct object
    
    Notes
    -----
    Given the highly-specific nature of the OutStruct matlab format, this
    function is mostly used internally by Neural Acoustic Processing
    Lab members.
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
            if f == 'resp' or f == 'aud':
                if tmp_t.ndim > 1:
                    tmp_t = tmp_t.transpose(1,0,*[i for i in range(2, tmp_t.ndim)]) # only switch the first 2 dimensions if there are more than 2
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
    
    out = OutStruct(data=data, strict=strict)
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
    to create an OutStruct object. The BIDS file structure is a commonly used structure
    for storing neural recordings such as EEG, MEG, or iEEG.
    
    The channels in the BIDS files are either stored in the 'resp' field of the
    OutStruct or the 'stim' field, depending on whether the `channel_type` is 'stim'.
    
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
        in the trial for the OutStruct. For example, if befaft=[1,1] then if each trial's
        recording is 10 seconds long, each trial in the resulting OutStruct will contain
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
        the OutStruct. By default, all channels which are not of type 'stim' will be included.
        Note, the order of these channels may not be conserved.
    
    Returns
    -------
    out : OutStruct
        Event/trial responses, stim, and other basic data in an OutStruct format.
        
    Notes
    -----
    The measurement information that is read-in by this function is stored in the outstruct.mne_info
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
    
    # build OutStruct
    outstruct_data = []
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
        outstruct_data.append(trial_data)  

    outstruct = OutStruct(outstruct_data, strict=False)
    outstruct.set_mne_info(raw_info)
    return outstruct
    
    
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
    
