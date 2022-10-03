from os.path import dirname, join

from .fileio import import_data
from ..data import Data

def load_speech_task_data():
    '''
    Load a sample Data object containing simulated intracranial EEG data
    from a speech task where a human subject listened to audiobook excerpts. The data
    contains 10 trials, each with a single-channel audio waveform, 10 simulated
    channels of electrodes, a transcript of the speech in the audio, and a
    128-channel auditory spectrogram.

    The electrode responses were simulated by adding noise to the predictions of
    10 different spectro-temporal receptive field models.

    Returns
    -------
    data : naplib.Data instance
        Task data containing 10 trials of stimuli, responses, and metadata for a
        simulated intracranial EEG recording.

    '''

    # read in data from file
    filedir = dirname(__file__)
    filepath = join(filedir, 'sample_data/demo_data.mat')

    return import_data(filepath, strict=False)
