import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import stats
from tqdm.auto import tqdm, trange

def align_stimulus_to_recording(rec_audio, rec_fs, stim_dict, stim_order,
 use_hilbert=True, confidence_threshold=0.2, t_use=None, t_search=120, t_start_look=0, verbose=True):
    '''
    Find the times that correspond to the start and end time of each stimulus

    Parameters
    ----------
    rec_audio: np.ndarray (time,)
        Recorded stimulus or triggers from hospital system
    rec_fs: int | float
        Sampling frequncy of recorded stimulus/triggers
    stim_dict: dict
        Dictionary in which keys are stimulus filenames and values are a tuple of
        (stimulus sampling frequency: int | float, stimulus waveform: np.ndarray (time, channels))
    stim_order: list of strings
        List of stimulus filenames in order of presentation
    use_hilbert: bool
        Whether or not to apply hilbert transform to audio for alignment
        By default, apply hilbert transform
    confidence_threshold: float [0,1]
        How large the Pearson's correlation between the recorded audio and the stimulus
        must be to consider the stimulus sufficiently detected
        By default, use 0.2
    t_use: float | None
        How much time of the stimulus to use for alignment
        Less time is less computationally intensive and will run quicker
        If None, uses the entire stimulus
    t_search: float
        How much time of the recorded audio to search on each try
        Less time is less computationally intensive and will run quicker
        By default, uses 30 seconds
    t_start_look: float
        What time of the recorded audio to begin the searching process
        By default, uses 0 seconds, i.e. beginning of recorded audio

    Returns
    -------
    alignment_inds: list of tuples
        Indices of stimulus alignment in which each list element corresponds to each trial in stim_order
        and each tuple contains the (start_index, end_index) of each trial, matched to rec_fs
    alignment_confidence: list of floats
        Pearson's correlation between the stimulus and recorded audio for each trial
    '''


    # Optionally, derive analytic envelope of the recorded audio
    if use_hilbert:
        rec_audio = np.abs(sig.hilbert(rec_audio))

    # Convert time-based arguments to indices
    n_start_look  = int(t_start_look * rec_fs)
    n_search = int(t_search * rec_fs)
    if t_use != None:
        n_use = int(t_use * rec_fs)
    
    # Instantiate empty list of alignments
    alignment_inds = []
    alignment_confidence = []
    useCh = None

    # Iterate over each stimulus provided in stim_order
    for stim_name in tqdm(stim_order):
        # Get stimulus waveform and sampling rate
        stim_fs, stim_full = stim_dict[stim_name]

        # If stimulus is multi-channel, align each channnel and then find best
        if len(stim_full.shape) > 1 and useCh == None:
            stims_curr = [stim_full[:,ch] for ch in range(stim_full.shape[1])]
        elif len(stim_full.shape) == 1:
            stims_curr = [stim_full]
        else:
            stims_curr = [stim_full[:,useCh]]

        # Resample stimulus to recorded audio sampling rate
        stims_ds = [sig.resample(stim, int(len(stim)/stim_fs*rec_fs)) for stim in stims_curr]

        # Save n_start_look for when checking multi-channel stimulus audio
        if len(alignment_inds) == 0:
            stim_start_look = n_start_look
        else:
            # End index of most recently found trial
            stim_start_look = alignment_inds[-1][1]

        possible_inds = []
        max_corrs = []

        # Align each channel, find best and set useCh on first iteration
        for stim in stims_ds:
            # Reset n_start_look
            n_start_look = stim_start_look

            # Truncate stimulus if using only partial
            if t_use == None:
                stim_use = stim
            else:
                stim_use = stim[:int(n_use * rec_fs)]

            # Get analytic envelope of stimulus
            if use_hilbert:
                stim_use = np.abs(sig.hilbert(stim_use))

            # Search recorded audio until correlation confidence threshold is met
            FOUND = False
            while not FOUND:
                # Get segment of recorded audio to search
                rec_audio_use = rec_audio[n_start_look: n_start_look + n_search]
                # Perform cross correlation
                corrs = sig.correlate(rec_audio_use, stim_use, mode='full')
                # Remove segment where stim is correlated with zeros (instead of recorded audio)
                corrs = np.abs(corrs[len(stim_use)-1:])

                # Find points of cross correlation to search
                corr_peaks = sig.find_peaks(corrs, height=np.percentile(corrs, 99.9))[0]

                # Find which peak produces the highest Pearson's correlation
                max_val = 0
                for pk in corr_peaks:
                    if n_start_look + pk + len(stim_use) < len(rec_audio):
                        curr_corr = stats.pearsonr(
                            stim_use,
                            rec_audio[n_start_look + pk : n_start_look + pk + len(stim_use)])[0]
                        if curr_corr > max_val:
                            max_val = curr_corr
                            max_ind = pk

                if verbose:
                    print(f'Started looking at t={n_start_look/rec_fs:.2f}')
                    print(f'Found segment with correlation={max_val:.4f}')

                if max_val > confidence_threshold:
                    FOUND = True

                    # Save possible alignment (start ind, end ind)
                    possible_inds.append((n_start_look + max_ind, n_start_look + max_ind + len(stim)))
                    max_corrs.append(max_val)

                    if verbose:
                        print(f'Found {stim_name}', possible_inds)
                else:
                    # Jump ahead by 1/10th of the search time
                    n_start_look = n_start_look + n_search//10


        # Determine which stimulus channel was best, set useCh, save inds
        if useCh == None:
            print(f'Using channel {np.argmax(max_corrs)}')
        useCh = np.argmax(max_corrs)
        alignment_inds.append(possible_inds[useCh])
        alignment_confidence.append(np.amax(max_corrs))


    return alignment_inds, alignment_confidence





