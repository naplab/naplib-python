import numpy as np
import os
import contextlib
import wave

from naplib.features import auditory_spectrogram, peak_rate

def test_compare_peakRate():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(curr_dir, 'test_aligner_files/wav_files', 'test1.wav')

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

    aud = auditory_spectrogram(sound_normalized, rate, frame_len=8, tc=4, factor='linear')
    pr = peak_rate(aud, 100)
    pr_inds = np.where(pr)[0]
    pr_vals = pr[pr_inds]

    pr_inds_truth = np.array([  5,  17,  32,  55,  82,  93, 180, 199, 234, 274, 375,
        411, 422, 502, 574, 608, 620, 633, 645])
    pr_vals_truth = np.array([0.01168778, 0.00851577, 1.16018359, 0.355288  , 0.0789882 ,
        0.04822385, 0.1053116 , 0.80346552, 0.58262492, 0.04549265,
        0.72904072, 0.05913313, 0.03868357, 0.00780381, 0.80212248,
        0.03482683, 0.07629197, 0.11645026, 0.02647678])


    assert np.allclose(pr_vals, pr_vals_truth, atol=1e-3)
    assert np.allclose(pr_inds, pr_inds_truth)

