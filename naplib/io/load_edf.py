import re
import array
import numpy as np
from math import ceil, floor
from typing import Iterable, Dict
from datetime import datetime


def load_edf(path: str, t1: float=0, t2: float=0) -> Dict:
    """
    Load data from EDF file (*.edf).

    Notes
    -----
    This function supports the original EDF format and EDF+C. For the EDF+D format
    it may be better to use PyEDFlib or mne.io.
    
    Parameters
    ----------
    path : str, path-like
        Path to EDF data file
    t1 : float, default=0
        Starting time to extract
    t2 : float, default=0
        Ending time to extract until (default of 0 extracts until end of file)
    
    Returns
    -------
    loaded_dict : dict from string to numpy array or float
        Keys: 'data' - loaded neural recording (time*channels), 'data_f' - sampling rate of data,
        'wav' - loaded audio recording (time*channels), wav_f' - sampling rate of sound,
        'labels_data' - array of labels for the channel streams,
        'labels_wav' - array of labels for the audio streams

    """

    with open(path, 'rb') as fin:
        # read header
        version = int(fin.read(8).decode('ascii'))
        patient = fin.read(80).decode('ascii').strip()
        recording = fin.read(80).decode('ascii').strip()
        start_date = fin.read(8).decode('ascii')
        start_time = fin.read(8).decode('ascii')
        header_size = int(fin.read(8).decode('ascii'))
        reserved = fin.read(44).decode('ascii').strip()

        # current implementation only supports contiguous EDF
        if version != 0:
            raise ValueError('EDF with version != 0 not supported.')
        if reserved not in ('', 'EDF+C'):
            raise ValueError('EDF reserved field should be empty (original EDF) or "EDF+C" (contiguous EDF+).')

        # number of lengths of signals
        num_records = int(fin.read(8).decode('ascii'))
        record_dur = float(fin.read(8).decode('ascii'))
        num_signals = int(fin.read(4).decode('ascii'))

        # channel metadata
        labels = np.array([fin.read(16).decode('ascii').strip() for _ in range(num_signals)])
        transducer = np.array([fin.read(80).decode('ascii').strip()  for _ in range(num_signals)])
        physical_dim = np.array([fin.read(8).decode('ascii').strip()  for _ in range(num_signals)])
        physical_min = np.array([float(fin.read(8).decode('ascii'))  for _ in range(num_signals)])
        physical_max = np.array([float(fin.read(8).decode('ascii'))  for _ in range(num_signals)])
        digital_min = np.array([float(fin.read(8).decode('ascii'))  for _ in range(num_signals)])
        digital_max = np.array([float(fin.read(8).decode('ascii'))  for _ in range(num_signals)])
        prefilt = np.array([fin.read(80).decode('ascii').strip()  for _ in range(num_signals)])
        samples = np.array([int(fin.read(8).decode('ascii'))  for _ in range(num_signals)])
        samples_per_record = samples.sum()
        fin.seek(32 * num_signals, 1)
        
        if labels[-1] == 'EDF Annotations':
            skip_end_bytes = samples[-1] * 2
            num_signals -= 1
            labels = labels[:-1]
            transducer = transducer[:-1]
            physical_dim = physical_dim[:-1]
            physical_min = physical_min[:-1]
            physical_max = physical_max[:-1]
            digital_min = digital_min[:-1]
            digital_max = digital_max[:-1]
            prefilt = prefilt[:-1]
            samples = samples[:-1]
        else:
            skip_end_bytes = 0

        if len(set(samples)) != 1:
            raise RuntimeError(
                'The load_edf function does not support heterogeneous `samples` per record, '
                'except for EDF annotations.'
            )
        samples = samples[0]

        # compute sampling rate
        sampling_rate = samples / record_dur

        # only process part of recording that falls between the specified time range (t1, t2)
        min_record = floor(t1 / record_dur)
        max_record = ceil(t2 / record_dur) if 0 < t2 < num_records*record_dur else num_records
        fin.seek(min_record * samples_per_record * 2, 1)
        num_records = max_record - min_record
        t_skip = min_record * record_dur
        
        # scale recordings from digital to physical units
        scale = (physical_max - physical_min) / (digital_max - digital_min)
        def rescale(x):
            return (x - digital_min) * scale + physical_min

        # read actual neural data
        aux_signals = _aux_channels(labels)
        num_aux_signals = sum(aux_signals)
        num_data_signals = num_signals - num_aux_signals
        data = np.zeros((num_records * samples, num_data_signals), dtype=np.float32)
        aux_data = np.zeros((num_records * samples, num_aux_signals), dtype=np.float32)
        for i in range(num_records):
            buffer = array.array('h')
            buffer.fromfile(fin, num_signals * samples)
            buffer = np.array(buffer).reshape(num_signals, samples).T
            buffer = rescale(buffer)
            data[i*samples:(i+1)*samples] = buffer[:, ~aux_signals]
            aux_data[i*samples:(i+1)*samples] = buffer[:, aux_signals]
            fin.seek(skip_end_bytes, 1)
    
    # Convert start_date and start_time to python objects
    try:
        start_date = datetime.strptime(start_date, '%m.%d.%y').date()
    except:
        start_date = None
    try:
        start_time = datetime.strptime(start_time, '%H.%M.%S').time()
    except:
        start_time = None

    return {
        'data': data,
        'data_f': sampling_rate,
        'wav': aux_data,
        'wav_f': sampling_rate,
        'labels_data': labels[~aux_signals],
        'labels_wav': labels[aux_signals],
        't_skip': t_skip,
        'info': {
            'start_date': start_date,
            'start_time': start_time,
            'patient': patient,
            'version': version,
            'transducer': transducer,
            'physical_unit': physical_dim,
            'prefilt': prefilt,
        },
    }


def _aux_channels(labels: Iterable[str]):
    pattern = r'^(EKG[LR]?|DC[0-9]*|TRIG[0-9]*|OSAT|PR|Pleth|EDF Annotations)$'
    return np.array([re.match(pattern, label) for label in labels], dtype=bool)

