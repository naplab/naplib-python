import os
import re
import numpy as np
from typing import Dict, Optional, Tuple, Set
from scipy.io import wavfile

def load_wav_dir(directory: str, pattern: Optional[str]=None, rescale: bool=True, subset: Optional[Set[str]]=None) -> Dict[str, Tuple[float, np.ndarray]]:
    """
    Load a set of wav files in a directory and return then in a dict mapping
    from filename (without the .wav suffix) to tuples of floats and numpy arrays
    containing the sampling rate and wav data.
    
    Parameters
    ----------
    directory : str, path-like
        Directory containing wav files. All wav files will be loaded and all other
        files will be ignored
    pattern : str, optional
        If provided, should be a regex pattern which will be used to match against
        the wav files found in the directory. For example, if ``pattern=r".*_stim.*",
        then only the wav files whose base name contains "_stim" will be loaded.
    rescale : bool, default=True
        If True, convert each input to a float in the range -1 to 1 based on the
        max value of the loaded dtype. For example, a wav file stored as 16-bit
        integers will be rescaled to np.float32 between -1 and 1 by dividing by
        32768.0. This is only done on wav files that are integer types. If True,
        output is always of type np.float32
    subset : Set[str], default=None
        If provided, only this subset of files will be loaded.
    
    Returns
    -------
    loaded_dict : dict from string to tuple of float (fs) and numpy array (wav data)
    """
    wav_files = [x for x in os.listdir(directory) if len(x) >= 4 and x[-4:]=='.wav']
    if subset is not None:
        wav_files = subset.copy()
    if pattern is not None:
        wav_files = [x for x in wav_files if re.match(pattern, x)]
    
    loaded_dict = {}
    
    for wav_name in wav_files:
        fs, data = wavfile.read(os.path.join(directory, wav_name))
        loaded_dict[wav_name] = (fs, data) # separated the tuple when reading file and inputting here for code-readability
        if rescale:
            # check dtype and only
            if data.dtype in [np.int16, np.int32]:
                dtype_info = np.iinfo(data.dtype)
                loaded_dict[wav_name] = (fs, (data / -np.float32(dtype_info.min)).astype(np.float32))
            elif  data.dtype in [np.uint8]:
                dtype_info = np.iinfo(data.dtype)
                loaded_dict[wav_name] = (fs, ((data / np.float32(128.0)) - 1).astype(np.float32)) # map between -1 and 1
            else:
                loaded_dict[wav_name] = (fs, data.astype(np.float32))
    
    return loaded_dict
