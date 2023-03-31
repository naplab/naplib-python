from tdt import read_block
import numpy as np
from typing import Dict, Optional
from scipy.signal import resample

def load_tdt(directory: str, t1: float=0, t2: float=0, wav_stream: str='Wav5') -> Dict:
    """
    Load data from TDT structure. Directory should contain .sev files and a .tev file, as
    well as other metadata files.
    
    Parameters
    ----------
    directory : str, path-like
        Directory containing TDT data files (tev, sev, and/or tin files, etc.)
    t1 : float, default=0
        Starting time to extract
    t2 : float, default=0
        Ending time to extract until (default of 0 extracts until end of file)
    wav_stream : str, default='Wav5'
        The name of the stream containing the audio recording to extract.
    
    Returns
    -------
    loaded_dict : dict from string to numpy array or float
        Keys: 'data' - loaded neural recording (time*channels), 'data_f' - sampling rate of data,
        'wav' - loaded audio recording (time*channels), wav_f' - sampling rate of sound,
        'labels' - array of labels for the channel streams
    """
    
    data = read_block(directory, t1=t1, t2=t2)
    
    eeg_data = []
    eeg_stream_labels = []
    
    streams = data['streams'].keys()
    
    if 'EEG1' in streams:
        eeg_data.append(data['streams']['EEG1']['data'])
        eeg_stream_labels.append(np.asarray(['EEG1']*eeg_data[0].shape[0]))
    elif 'RAWx' in streams:
        eeg_data.append(data['streams']['RAWx']['data'])
        eeg_stream_labels.append(np.asarray(['RAWx']*eeg_data[0].shape[0]))
    else:
        raise ValueError(f'Neither EEG1 nor RAWx streams present in TDT data.')
        
    min_len = eeg_data[0].shape[1]
    
    if 'EEG2' in streams:
        if data['streams']['EEG1']['fs'] != data['streams']['EEG2']['fs']:
            desired_len = data['streams']['EEG1']['fs']/data['streams']['EEG2']['fs'] * data['streams']['EEG2']['data'].shape[1]
            eeg2_data = resample(data['streams']['EEG2']['data'], desired_len, axis=1)
        else:
            eeg2_data = data['streams']['EEG2']['data']
            
        eeg_data.append(eeg2_data)
        eeg_stream_labels.append(np.asarray(['EEG2']*eeg_data[1].shape[0]))
        
        if eeg_data[1].shape[1] != eeg_data[0].shape[1]:
            min_len = min([min_len, eeg_data[1].shape[1]])
            eeg_data = [x[:,:min_len] for x in eeg_data]
            
    loaded_dict = {}
    loaded_dict['data'] = np.concatenate(eeg_data, axis=0).T
    loaded_dict['data_f'] = data['streams']['EEG1']['fs']
    loaded_dict['wav'] = data['streams'][wav_stream]['data'].T
    loaded_dict['wav_f'] = data['streams'][wav_stream]['fs']
    loaded_dict['labels_data'] = np.concatenate(eeg_stream_labels)
    loaded_dict['labels_wav'] = np.array([f'Wav5_{i}' for i in range(loaded_dict['wav'].shape[1])])
    
    return loaded_dict
