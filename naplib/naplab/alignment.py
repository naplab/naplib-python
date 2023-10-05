import numpy as np
import scipy.signal as sig
from scipy import stats
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import logging

from naplib import logger

def align_stimulus_to_recording(rec_audio, rec_fs, stim_dict, stim_order,
    use_hilbert=True, confidence_threshold=0.2, t_search=120, t_start_look=0):
    '''
    Find the times that correspond to the start and end time of each stimulus.

    Parameters
    ----------
    rec_audio : np.ndarray of shape (time,)
        Recorded stimulus or triggers from hospital system
    rec_fs : int | float
        Sampling frequncy of recorded stimulus/triggers
    stim_dict : dict
        Dictionary in which keys are stimulus filenames and values are a tuple of
        (stimulus sampling frequency: int | float, stimulus waveform: np.ndarray (time, channels))
    stim_order : list of strings
        List of stimulus filenames in order of presentation
    use_hilbert : bool, default=True
        Whether or not to apply hilbert transform to audio for alignment
    confidence_threshold : float in range [0,1], default=0.2
        How large the Pearson's correlation between the recorded audio and the stimulus
        must be to consider the stimulus sufficiently detected
    t_search : float, default=120
        How much time of the recorded audio to search on each try.
        Increase for long breaks between trials (or for started,stoped,restarted trials)
    t_start_look : float, default=0
        What time of the recorded audio to begin the searching process.
        By default, uses 0 seconds, i.e. beginning of recorded audio

    Returns
    -------
    alignment_times: list of tuples
        Indices of stimulus alignment in which each list element corresponds to each trial in stim_order
        and each tuple contains the (start_time, end_time) of each trial
    alignment_confidence: list of floats
        Pearson's correlation between the stimulus and recorded audio for each trial
    '''

    # Z-score the recorded audio to discard DC offset
    rec_audio = stats.zscore(rec_audio, axis=0)

    # Optionally, derive analytic envelope of the recorded audio
    if use_hilbert:
        rec_audio = np.abs(sig.hilbert(rec_audio))

    # Convert time-based arguments to indices
    n_start_look  = int(t_start_look * rec_fs)
    n_search = int(t_search * rec_fs)
    
    # Instantiate empty list of alignments
    alignment_times = []
    alignment_confidence = []
    useCh = None

    # Iterate over each stimulus provided in stim_order
    for stim_name in tqdm(stim_order, total=len(stim_order), disable=not logger.isEnabledFor(logging.INFO)):
        # Get stimulus waveform and sampling rate
        stim_fs, stim_full = stim_dict[stim_name]

        # If stimulus is multi-channel, align each channnel and then find best

        if stim_full.ndim > 1 and stim_full.shape[1] == 1:
            stim_full = stim_full.squeeze(1)

        if stim_full.ndim == 1:
            stims_curr, useCh = [stim_full], 0
        elif stim_full.ndim > 1 and useCh is None:
            stims_curr = [stim_full[:,ch] for ch in range(stim_full.shape[1])]
        else:
            stims_curr = [stim_full[:,useCh]]

        # Resample stimulus to recorded audio sampling rate
        stims_ds = [sig.resample(stim, int(len(stim)/stim_fs*rec_fs)) for stim in stims_curr]

        # Save n_start_look for when checking multi-channel stimulus audio
        if len(alignment_times) == 0:
            stim_start_look = n_start_look
        else:
            # End index of most recently found trial
            stim_start_look = int(alignment_times[-1][1]*rec_fs)

        possible_times = []
        max_corrs = []

        # Align each channel, find best and set useCh on first iteration
        for stim in stims_ds:
            # Reset n_start_look
            n_start_look = stim_start_look

            # Get analytic envelope of stimulus
            if use_hilbert:
                stim = np.abs(sig.hilbert(stim))

            # Search recorded audio until correlation confidence threshold is met
            FOUND = False
            while not FOUND and n_start_look + len(stim) <= len(rec_audio):
                # Number of samples of recorded audio to search over
                n_search_use = min(n_search + len(stim), len(rec_audio) - n_start_look)

                # Get segment of recorded audio to search
                rec_audio_use = rec_audio[n_start_look : n_start_look + n_search_use]

                # Perform cross correlation
                corrs = sig.correlate(rec_audio_use, stim, mode='valid')

                # Find points of cross correlation to search
                corr_peaks = sig.find_peaks(corrs, height=np.percentile(corrs, 99.9))[0]

                # Find which peak produces the highest Pearson's correlation
                max_val = 0
                for pk in corr_peaks:
                    if n_start_look + pk + len(stim) < len(rec_audio):
                        curr_corr = stats.pearsonr(
                            stim,
                            rec_audio[n_start_look + pk : n_start_look + pk + len(stim)])[0]
                        if curr_corr > max_val:
                            max_val = curr_corr
                            max_ind = pk

                logger.debug(f'Searching from t={n_start_look/rec_fs:.2f}, found segment with correlation={max_val:.4f}')

                if max_val > confidence_threshold:
                    FOUND = True

                    # Save possible alignment (start time, end time)
                    possible_times.append((
                        (n_start_look + max_ind)/rec_fs,
                        (n_start_look + max_ind + len(stim))/rec_fs
                        ))
                    max_corrs.append(max_val)

                    # Jump ahead to 99th percentile of the found stimulus
                    n_start_look = n_start_look + max_ind + len(stim)*99//100
                else:
                    # Jump ahead to 99th percentile of last search window
                    n_start_look = n_start_look + n_search*99//100

            if not FOUND:
                plt.figure()
                plt.plot(rec_audio)
                plt.show()
                raise ValueError(f'Failed to find stimulus {stim_name} during alignment.')

        # Determine which stimulus channel was best, set useCh, save inds
        if useCh is None:
            useCh = np.nanargmax(max_corrs)
            logger.info(f'Using stimulus channel {useCh} for alignment')
            alignment_times.append(possible_times[useCh])
            alignment_confidence.append(max_corrs[useCh])
        else:
            alignment_times.append(possible_times[0])
            alignment_confidence.append(max_corrs[0])
        
        logger.info(f'Found {stim_name} with correlation={alignment_confidence[-1]:.4f} @ {alignment_times[-1]}')

    return alignment_times, alignment_confidence
