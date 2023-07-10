from typing import List
import os
import numpy as np
import pathlib

def _load_eeg_locs(fname):
    """
    Helper function to load .locs file for EEG electrode locations.
    
    Parameters
    ----------
    fname : str, pathlike
        Path to file

    Returns
    -------
    pos : np.ndarray
        2D positions, shape (n_channels, 2)
    names : List[str]
        Name of each channel, length is n_channels
    """
    
    with open(fname, 'r') as infile:
        lines = [x.strip().split('\t') for x in infile.readlines()]
        
    pos = []
    names = []
        
    for i in range(len(lines)):
        line = [x.replace(" ", "") for x in lines[i]]
        pos.append([float(line[1]), float(line[2])])
        names.append(line[-1])
        
    pos = np.asarray(pos)
    
    return pos, names
    
def eeg_locs(setup='gtec62'):
    """
    Load EEG cap positions for a gtec locs file. By default, loads the
    gtec 62-channel EEG setup.
    
    These can be put into mne.viz.plot_topomap
    
    Parameters
    ----------
    setup : str or pathlike, default='gtec62'
        The setup to use. By default, uses the gtec62.locs file in the same directory.
        If another string or path is specified, that is assumed to be a path to
        a .locs file, and this will try to load that as an EEGLab .locs file. The coordinates
        in such a file must be polar coordinates with a head disk radius of 0.5.
    
    Returns
    -------
    pos : np.ndarray
        2D positions, shape (n_channels, 2)
    
    Examples
    --------
    >>> import mne
    >>> import numpy as np
    >>> from naplib.visualization import eeg_locs
    >>> arr = np.random.rand(62,)
    >>> # using the default locs
    >>> mne.viz.plot_topomap(arr, eeg_locs())
    >>> # using a custom .locs file in the current directory
    >>> mne.viz.plot_topomap(arr, eeg_locs('custom.locs'))
    """

    if setup == 'gtec62':
        curr_dir = pathlib.Path(__file__).parent.resolve()
        pos, _ = _load_eeg_locs(os.path.join(curr_dir, 'gtec62.locs'))
    else:
        pos, _ = _load_eeg_locs(setup)

    def pol2cart(rho, phi):
        '''0 degrees is central North, and then positive angles are in clockwise direction'''
        x = rho * np.cos(-(phi-90) * np.pi/180)
        y = rho * np.sin(-(phi-90) * np.pi/180)
        return np.asarray([x, y]).T / (.5/.095) # convert from head disk radius of .5 to 0.095 (EEGLab to MNE)

    return pol2cart(pos[:,1], pos[:,0])
