"""
==========================
Phoneme and Word Alignment
==========================

Perform forced alignment for words and phonemes.

In this notebook we demonstrate how to perform alignment between audio and text transcripts of a series of trials.
"""

import numpy as np
import matplotlib.pyplot as plt

import naplib as nl
from naplib.features import Aligner
from naplib.segmentation import segment_around_label_transitions

data = nl.io.load_speech_task_data()

###############################################################################
# Full Alignment on Data Instance
# -------------------------------
# 
# In this case, all sounds and transcripts as well as several other metadata fields must be present in the Data.
# They will be aligned and results will be given in a new Data instance of label vectors.

# Specify the directory to store the output files from alignment (.TextGrid, .phn, and .wrd files)
output_dir = './alignment_output_data2'

aligner = Aligner(output_dir, tmp_dir='data2_/')

# the Aligner's align() method requires a length field which specifies the number of
# samples that we want each resulting label vector to contain
data['length'] = [x.shape[0] for x in data['resp']]

# Two of the other critical fields we need for alignment include the 'sound', and the 'transcript'
print(f'The "sound" field is type {type(data[0]["sound"])} and the first trial is shape {data[0]["sound"].shape}')

# This is the first trial's 'transcript' (just the first 100 characters), the words spoken in the trial's stimulus
print(data[0]["script"][:100])


# In this Data instance, the transcript is in a field called "script", not the default "transcript",
# so we need to specify that.
# If other required fields were named differently from their default values, we would need
# to specify those as well. See documentation for details of required fields.
result = aligner.align(data, transcript='script') # perform alignment


###############################################################################
# Alternative: Perform Alignment from Directories with Stimulus Files
# -------------------------------------------------------------------
# 
# If we have the sounds and transcripts in files but not in our Data instance, we can put all sounds (.wav files) in one directory and all matching transcripts (.txt files) in another directory.
# 
# 
# audio_dir = './sounds'
# text_dir = './scripts'
# output_dir = './alignment_output'

# Create aligner
# aligner = Aligner(output_dir, tmp_dir='data_/')

# Align files
# aligner.align_files(audio_dir, text_dir)

# Get resulting labels
# result = aligner.get_label_vecs_from_files(data)


###############################################################################
# Visualize Alignment
# -------------------
# 
# Now, we are left with a Data object containing fields with the alignment information.
# We can visualize the label vectors for phonemes, manner of articulation, and words over
# time for a few trials. For each trial, the sound consists of a beep followed by 4 words.


# Let's just look at the first 10 seconds of the first 2 trials of label vectors aligned to the stimulus
for n, trial in enumerate(data[:2]):
    
    print(f'Trial {n}')
    
    fig, axes = plt.subplots(4,1,figsize=(9,6), sharex=True, gridspec_kw={'height_ratios':[2,1,1,1]})
    
    axes[0].imshow(trial['aud'][:1000].T**.3, origin='lower', aspect=1)
    axes[0].set_title('Spectrogram')
    
    axes[1].plot(result['phn_labels'][n][:1000])
    axes[1].set_title('Phoneme Labels')
    
    axes[2].plot(result['manner_labels'][n][:1000])
    axes[2].set_title('Manner of Articulation Labels')
    
    axes[3].plot(result['wrd_labels'][n][:1000])
    axes[3].set_title('Word Labels')
    
    plt.tight_layout()
    
    plt.show()


###############################################################################
# Look at ERPs from Onsets of Words and Phonemes
# ----------------------------------------------
# 
# One simple thing we can look at using these label vectors is the neural response to word onsets.


segments, labels, prior_labels = segment_around_label_transitions(data, field='resp', labels=result['wrd_labels'], prechange_samples=50, postchange_samples=200)
segments, labels, prior_labels = segment_around_label_transitions(data, field='resp', labels=result['phn_labels'], prechange_samples=50, postchange_samples=200)

print(segments.shape) # num_transitions, time, num_electrodes

print(labels.shape) # num_transitions
print(prior_labels.shape) # num_transitions


# plot the average phoneme-onset response for each electrode
time_vec = np.linspace(-0.5, 2, 250)
plt.figure()
plt.plot(time_vec, segments.mean(0))
plt.show()


###############################################################################
# Compute Electrode Lags by Phoneme F-Ratio
# -----------------------------------------
# 
# Now that we have the alignment, one thing we can do is compute the electrode lags
# as measured by the peak in their f-ratio to phonemes.


from naplib.segmentation import electrode_lags_fratio

lags, fratios = electrode_lags_fratio(data, field='resp', labels=result['phn_labels'], max_lag=30, return_fratios=True)
fratios.shape

# We can see that the peaks in the f-ratio in separating phoneme responses is typically
# between about 100 and 200 ms after the onset of the phoneme
print(lags) # electrode lags, in samples

plt.figure()
plt.plot(fratios.T)
plt.yscale('log')
plt.ylabel('F-ratio')
plt.xlabel('Samples since phoneme change')
plt.show()

