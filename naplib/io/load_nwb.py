from typing import Dict
import numpy as np


def load_nwb(filepath: str) -> Dict:
    """
    Load data from NWB structure. File should be string path to a .nwb file
    
    Parameters
    ----------
    filepath : str, path-like
        Path to NWB data file.

    Returns
    -------
    loaded_dict : dict from string to numpy array or float
        Keys: 'data' - loaded neural recording (time*channels), 'data_f' - sampling rate of data,
        'wav' - loaded audio recording (time*channels), wav_f' - sampling rate of sound,
        'labels' - array of labels for the channel streams
    """

    try:
        from pynwb import NWBHDF5IO
    except Exception:
        raise Exception('Missing package pynwb which is required for loading data from NWB format. Please '
            'install it with "pip install pynwb"')

    with NWBHDF5IO(filepath, "r") as reader:
        loaded_data = reader.read()

        loaded_dict = {}
        loaded_dict['data'] = loaded_data.acquisition['ieeg'].data[:]
        loaded_dict['data_f'] = loaded_data.acquisition['ieeg'].rate
        loaded_dict['wav'] = loaded_data.acquisition['audio'].data[:]
        loaded_dict['wav_f'] = loaded_data.acquisition['audio'].rate
        loaded_dict['labels_data'] = np.array(['NWB1'] * loaded_dict['data'].shape[1])
        loaded_dict['labels_wav'] = np.array([f'audio_{i}' for i in range(loaded_dict['wav'].shape[1])])

    return loaded_dict
