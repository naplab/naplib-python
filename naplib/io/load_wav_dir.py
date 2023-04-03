import os
import re
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.io import wavfile

def load_wav_dir(directory: str, pattern: Optional[str]=None) -> Dict[str, Tuple[float, np.ndarray]]:
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
    
    Returns
    -------
    loaded_dict : dict from string to tuple of float (fs) and numpy array (wav data)
    """
    wav_files = [x for x in os.listdir(directory) if len(x) >= 4 and x[-4:]=='.wav']
    if pattern is not None:
        wav_files = [x for x in wav_files if re.match(pattern, x)]
    
    loaded_dict = {}
    
    for wav_name in wav_files:
        fs, data = wavfile.read(os.path.join(directory, wav_name))
        loaded_dict[wav_name] = (fs, data) # separated the tuple when reading file and inputting here for code-readability
    
    return loaded_dict
