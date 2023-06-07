"""Continuous neural data (CND) format used for mTRF-Toolbox"""
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Union
from collections import defaultdict

import mne
import numpy as np
from hdf5storage import loadmat

from naplib import logger
from naplib.data import concat, Data


def load_cnd(
    filepath: str,
    load_stims: Union[bool, str] = True,
    truncate_lengths: bool = True,
    connectivity: Optional[Union[str, Sequence, float]] = None,
):
    """Load continuous neural data (CND) file used in the mTRF-Toolbox.

    Parameters
    ----------
    filepath : str
        Path to the data file (``*.mat``). This can be either the `stim` data or
        the `eeg` data.
    load_stims : Union[bool, str], default=True
        If True (default), try to load stimuli from an inferred filepath by looking for `dataStimXX.mat`,
        where XX is the subject number parsed from filepath, or fall back on `dataStim.mat`,
        under the same directory as `filepath`. Optionally, the exact path to the stim file
        can be specified. If False, only the file specified by `filepath` is
        loaded. This argument is ignored if `stim` is contained in the data loaded
        from `filepath`.
    truncate_lengths : bool, default=True
        If True, and there are both `eeg` and `stim` data loaded, truncate the lengths of
        the `eeg` and all the stimuli to match each other. The beginnings of all features
        and `eeg` are assumed to be aligned, and the end are truncated to the same length on
        a trial-by-trial basis.
    connectivity : Optional[Union[str, Sequence, float]], default=1.6
        Sensor adjacency graph for EEG sensors.
        By default, the function tries to use the ``deviceName`` entry and falls
        back on distance-based connectivity for unknown devices.
        Can be explicitly specified as a `FieldTrip neighbor file
        <https://www.fieldtriptoolbox.org/template/neighbours/>`_ (e.g., ``'biosemi64'``;
        Use a `float` for distance-based connectivity. This connectivity info will be
        put into the `info` attribute of the naplib.Data instance returned.

    Returns
    -------
    data : Data
        Data containing the various trials loaded from the file, as well as all associated
        metadata for each trial. Some metadata, including connectivity, is located in the
        `info` attribute of the Data object.

    Notes
    -----
    If stimuli and eeg are not the same length, it will be assumed that they

    This loading function is modified from the `read_cnd` function found in
    `Eelbrain<https://eelbrain.readthedocs.io/en/stable/index.html>`_
    """
    path = Path(filepath)
    if not path.suffix and not path.exists():
        path = path.with_suffix('.mat')

    data = loadmat(str(path), simplify_cells=True)

    if 'stim' not in data and 'eeg' not in data:
        raise ValueError("File contains neither 'eeg' or 'stim' entry")

    data_eeg = {}
    info_dict = {}
    if 'eeg' in data:
        data_eeg =data['eeg']

        data_eeg['eeg'] = [x for x in data_eeg['data'].squeeze()]
        data_eeg.pop('data')

        data_eeg['fs'] = [data_eeg['fs'] for _ in data_eeg['eeg']]

        # EEG sensor properties
        dist_connectivity = None
        sysname = data_eeg.get('deviceName', None)
        chanlocs_info = None
        if 'chanlocs' in data_eeg:
            chanlocs_info = defaultdict(list)

            for touples in zip(*[d.items() for d in data_eeg['chanlocs']]):
                values = [tpl[1] for tpl in touples]
                chanlocs_info[touples[0][0]].extend(values)
            chanlocs_info = dict(chanlocs_info)

            ch_names = chanlocs_info['labels']
            chanlocs_info['XYZ'] = np.vstack([
                -np.array(chanlocs_info['Y']),
                chanlocs_info['X'],
                chanlocs_info['Z'],
            ]).T
            data_eeg.pop('chanlocs')

        # find connectivity
        if not connectivity:
            connectivity = 'none'
        elif isinstance(connectivity, str) and connectivity not in ('grid', 'none'):
            adj_matrix, adj_names = mne.channels.read_ch_adjacency(connectivity)
            # fix channel order
            if chanlocs_info is None:
                raise ValueError(
                    f'No channel loc information found in file, so cannot compute connectivity for device.')
            if adj_names != ch_names:
                index = np.array([adj_names.index(name) for name in ch_names])
                adj_matrix = adj_matrix[index][:, index]
            connectivity = _matrix_graph(adj_matrix)

        info_dict = {'connectivity': connectivity, 'chanlocs': chanlocs_info}

        if 'origTrialPosition' in data_eeg:
            orig_trial_position = data_eeg['origTrialPosition'].squeeze()
            if len(orig_trial_position) != len(data_eeg['eeg']):
                logger.warning(f"Ignoring origTrialPosition because it has the wrong length: {orig_trial_position!r}")
            else:
                data_eeg['origTrialPosition'] = list(orig_trial_position - 1)  # convert to zero-indexing
        else:
            logger.warning(f"origTrialPosition missing")

        # Extra channels
        if 'extChan' in data_eeg:
            data_eeg['extChan'] = data_eeg['extChan']['data']
        if 'reRef' in data_eeg:
            if type(data_eeg['reRef']) is str:
                info_dict['reRef'] = [data_eeg['reRef'] for _ in data_eeg['eeg']]
            else:
                info_dict['reRef'] = data_eeg['reRef'].squeeze()
            data_eeg.pop('reRef')

        # Add any other fields present
        for field in list(data_eeg.keys()):
            if not isinstance(data_eeg[field], list) or len(data_eeg[field]) != len(data_eeg['eeg']):
                try:
                    if len(data_eeg[field]) == len(data_eeg['eeg']):
                        data_eeg[field] = [x for x in data_eeg[field]]
                    else:
                        data_eeg[field] = [data_eeg[field] for _ in data_eeg['eeg']]
                except TypeError:
                    data_eeg[field] = [data_eeg[field] for _ in data_eeg['eeg']]

    # load stimuli

    data_stim = {}
    stim_names = []

    if 'stim' in data:
        data_stim, stim_names = _organize_stims(data)
    elif load_stims:
        if load_stims == True:
            # check if there is a file called dataStimXX.mat with the matching subject number first
            parsed_number = ''
            if 'eeg' in data:
                parsed_numbers = re.findall(r'\d+', str(path))
                if len(parsed_numbers) > 0:
                    parsed_number = parsed_numbers[-1]
            subj_specific_stim = os.path.join(path.parent.absolute(), f'dataStim{parsed_number}.mat')
            fall_back_stim = os.path.join(path.parent.absolute(), 'dataStim.mat')
            if os.path.exists(subj_specific_stim):
                load_stims = subj_specific_stim
            elif os.path.exists(fall_back_stim):
                load_stims = fall_back_stim
            else:
                raise ValueError(f'Tried to infer path to stimuli, since load_stims is True, but neither inferred '
                                 f'file path was not found:\n'
                                 f'{subj_specific_stim}\n'
                                 f'{fall_back_stim}')

            # load the stim file that we inferred
            logger.debug(f"Inferred stim filepath: {load_stims}")
            if not os.path.exists(load_stims):
                raise ValueError(
                    f'Tried to infer path to stimuli, since load_stims is True, but inferred file path was not found: {load_stims}')
        elif not isinstance(load_stims, str):
            raise TypeError(
                f"load_stims is not False, but must otherwise be True or a string path to a file, but got type {type(load_stims)}")

        # load_stims should be a string file path now, so try to load it
        logger.debug(f'Loading stimuli file: {load_stims}')
        data_stim = loadmat(load_stims, simplify_cells=True)
        if 'stim' not in data_stim:
            raise ValueError(f'"stim" variable not present in loaded stim data file {load_stims}')
        data_stim, stim_names = _organize_stims(data_stim)

    # convert to Data objects and put together

    data_eeg = Data(data_eeg)
    data_eeg.set_info(info_dict)

    data_stim = Data(data_stim)

    if len(data_eeg) > 0 and len(data_stim) > 0:
        concat_data = concat([data_eeg, data_stim], axis=1)
        # truncate eeg and stimuli so they are the same length
        if truncate_lengths:
            fields_to_truncate = ['eeg'] + stim_names
            for trial in concat_data:
                min_len = min([trial[f].shape[0] for f in fields_to_truncate])
                for f in fields_to_truncate:
                    trial[f] = trial[f][:min_len]
        return concat_data

    elif len(data_eeg) > 0:
        return data_eeg

    elif len(data_stim) > 0:
        return data_stim

    else:
        raise ValueError(f'No data was found in either `stim` or `eeg` variables in any file.')


def _organize_stims(data):
    """
    Organize stim data.
    """
    output = {}
    try:
        data_stim = {k: data['stim'][0][i].squeeze() for i, k in enumerate(data['stim'].dtype.names)}
        stim_names = [x[0,0] for x in data_stim['names'].squeeze()]
    except:
        data_stim = data['stim']
        stim_names = [x for x in data_stim['names'].squeeze()]
    output['stimIdxs'] = list(data_stim['stimIdxs'].squeeze() - 1)  # convert to zero-indexing
    stim_arrays = [list(x) for x in data_stim['data'].squeeze()]
    for name, arr in zip(stim_names, stim_arrays):
        output[name] = arr
    if 'condIdxs' in data_stim:
        output['condIdxs'] = [x for x in data_stim['condIdxs'].squeeze()]
    if 'fs' in data_stim:
        output['fs_stim'] = [data_stim['fs'] if type(data_stim['fs']) is int else data_stim['fs'].item() for _ in output['stimIdxs']]

    return output, stim_names


def _matrix_graph(matrix):
    """Copyright Christian Brodbeck 2017
    From Eelbrain
    Create connectivity from matrix"""
    coo = matrix.tocoo()
    assert np.all(coo.data)
    edges = {(min(a, b), max(a, b)) for a, b in zip(coo.col, coo.row) if a != b}
    return np.array(sorted(edges), np.uint32)
