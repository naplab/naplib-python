from tdt import read_block
import numpy as np
from typing import Dict

from naplib import logger


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
        'labels_data' - array of labels for the channel streams,
        'labels_wav' - array of labels for the audio streams
    """
    
    data = read_block(directory, t1=t1, t2=t2)
    streams = data['streams'] if 'streams' in data.keys() else data
    
    eeg_streams = [
        s for s in ('EEG1', 'EEG2', 'RAWx', 'RSn1', 'RSn2')
        if s in streams.keys() and 'data' in streams[s].keys()
    ]
    if not eeg_streams:
        raise ValueError(f'Neither of EEG1, EEG2 or RAWx streams present in TDT data.')
    
    eeg_data = []
    eeg_stream_fs = []
    eeg_stream_labels = []
    
    for i, s in enumerate(eeg_streams):
        eeg_data.append(streams[s]['data'])
        eeg_stream_fs.append(streams[s]['fs'])
        eeg_stream_labels.append(np.asarray([s] * streams[s]['data'].shape[0]))
    
    if len(np.unique(eeg_stream_fs)) != 1:
        raise ValueError(f'Found different sampling rates for EEG streams: {eeg_stream_fs}.')
    
    lengths = [x.shape[1] for x in eeg_data]
    if len(np.unique(lengths)) != 1:
        logger.warn(f'EEG streams have different lengths: {lengths}; clipping all to length of shortest stream.')
        
        min_len = min(lengths)
        for i in range(len(eeg_data)):
            if eeg_data[i].shape[1] != min_len:
                eeg_data[i] = eeg_data[i][:, :min_len]
    
    eeg_data = np.concatenate(eeg_data, axis=0).T
    eeg_stream_fs = eeg_stream_fs[0]
    eeg_stream_labels = np.concatenate(eeg_stream_labels)
    wav_data = streams[wav_stream]['data'].T
    wav_stream_fs = streams[wav_stream]['fs']
    wav_stream_labels = np.array([f'Wav5_{i}' for i in range(wav_data.shape[1])])
    
    info = {}
    if 'info' in data.keys():
        info['start_date'] = data['info']['start_date'].date()
        info['start_time'] = data['info']['start_date'].time()
    
    return {
        'data': eeg_data,
        'data_f': eeg_stream_fs,
        'wav': wav_data,
        'wav_f': wav_stream_fs,
        'labels_data': eeg_stream_labels,
        'labels_wav': wav_stream_labels,
        't_skip': t1,
        'info': info,
    }
