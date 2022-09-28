import os
import pytest
import numpy as np
import wave
import contextlib
import subprocess

from naplib.features import Aligner
from naplib import Data

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

    outstruct = Data([{'name': 'test1', 'sound': sound_normalized,
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
    aligner = Aligner(output_dir=dirs['out']+'1', tmp_dir=dirs['tmp']+'1', verbose=0)
    label_out = aligner.get_label_vecs_from_files(name=['test1'], dataf=[100], length=[dirs['duration']], befaft=[np.array([0,0])])
    assert isinstance(label_out, Data)
    for field in label_out.fields:
        for trial in label_out[field]:
            if isinstance(trial, np.ndarray):
                assert trial.shape[0] == dirs['duration']

def test_HTK_installation_check(dirs):
    try:
        subprocess.run('HLEd', check=True, capture_output=True)
        # HTK is installed, so do alignment and make sure it doesn't raise errors
        aligner = Aligner(output_dir=dirs['out']+'1', tmp_dir=dirs['tmp']+'1', verbose=0)
        aligner.align(data=dirs['outstruct'])
        assert True
    except (OSError, subprocess.SubprocessError, subprocess.CalledProcessError, FileNotFoundError):
        # HTK is not installed
        with pytest.raises(RuntimeError) as exc:
            aligner = Aligner(output_dir=dirs['out']+'1', tmp_dir=dirs['tmp']+'1', verbose=0)
            aligner.align(data=dirs['outstruct'])
        assert 'HTK may not be installed' in str(exc.value)

def test_alignment_from_outstruct(dirs):
    aligner = Aligner(output_dir=dirs['out']+'2', tmp_dir=dirs['tmp']+'2', verbose=0)
    label_out = aligner.get_label_vecs_from_files(data=dirs['outstruct'])
    for field in label_out.fields:
        for trial in label_out[field]:
            if isinstance(trial, np.ndarray):
                assert trial.shape[0] == dirs['outstruct']['resp'][0].shape[0]
