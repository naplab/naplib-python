import os
import pytest
import numpy as np
import wave
import contextlib

from naplib.features import Aligner
from naplib import OutStruct

@pytest.fixture(scope='module')
def dirs():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(curr_dir, 'test_aligner_files/wav_files', 'test1.wav')
    duration = None
    sound_normalized = None
    rate = None
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        sound = f.readframes(frames)
        # Convert buffer to float32 using numpy                                                                                 
        sound_int16 = np.frombuffer(sound, dtype=np.int16)
        sound_float32 = sound_int16.astype(np.float32).squeeze()

        # Normalise float32 array so that values are between -1.0 and +1.0                                                      
        max_int16 = 2**15
        sound_normalized = sound_float32 / max_int16

    outstruct = OutStruct([{'name': 'test1', 'sound': sound_normalized,
                            'soundf': rate, 'resp': np.random.rand(int(100*duration+200),5),
                            'dataf': 100, 'befaft': np.array([1.,1.]),
                            'length': int(100*duration+200),
                            'transcript': 'STOMACH SHOULDER HIP WAIST'}])

    return {'out': os.path.join(curr_dir, 'aligner_output'),
            'tmp': os.path.join(curr_dir, 'aligner_tmp'),
            'audio': os.path.join(curr_dir, 'test_aligner_files/wav_files'),
            'txt': os.path.join(curr_dir, 'test_aligner_files/txt_files'),
            'duration': int(100*duration),
            'outstruct': outstruct}
    

def test_alignment_from_directory(dirs):
    aligner = Aligner(output_dir=dirs['out'], tmp_dir=dirs['tmp'], verbose=0)
    aligner.align_files(dirs['audio'], dirs['txt'])
    label_out = aligner.get_label_vecs_from_files(name=['test1'], dataf=[100], length=[dirs['duration']], befaft=[np.array([0,0])])
    assert isinstance(label_out, OutStruct)
    assert label_out['phn_labels'][0].shape[0] == dirs['duration']


def test_alignment_from_outstruct(dirs):
    aligner = Aligner(output_dir=dirs['out'], tmp_dir=dirs['tmp'], verbose=0)
    aligner.align(outstruct=dirs['outstruct'])
    label_out = aligner.get_label_vecs_from_files(outstruct=dirs['outstruct'])
    assert label_out['wrd_labels'][0].shape[0] == dirs['outstruct']['resp'][0].shape[0]

def test_alignment_from_outstruct_matches_from_directory(dirs):
    aligner = Aligner(output_dir=dirs['out'], tmp_dir=dirs['tmp'], verbose=0)
    aligner.align(outstruct=dirs['outstruct'])
    label_outstruct = aligner.get_label_vecs_from_files(outstruct=dirs['outstruct'])

    aligner2 = Aligner(output_dir=dirs['out'], tmp_dir=dirs['tmp'], verbose=0)
    aligner2.align_files(dirs['audio'], dirs['txt'])
    label_dir = aligner2.get_label_vecs_from_files(name=['test1'], dataf=[100], length=[dirs['duration']], befaft=[np.array([0,0])])

    for field in label_outstruct.fields:
        for trial_outstruct, trial_dir in zip(label_outstruct[field], label_dir[field]):
            if isinstance(trial_outstruct, np.ndarray):
                print(field)
                print(trial_outstruct[100:-100] - trial_dir)
                assert np.allclose(trial_outstruct[100:-100], trial_dir)
