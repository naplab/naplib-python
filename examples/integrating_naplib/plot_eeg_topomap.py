"""
==================================================
Plotting EEG Topomap of Alpha/Theta Ratio with MNE
==================================================

Basic STRF fitting tutorial.

MNE is a popular python toolbox for analyzing neural data,
and it has a lot of visualization capabilities. In this tutorial, we show how to interface
between `naplib-python` and `mne` to produce EEG topomaps.
"""
# Author: Gavin Mischler
# 
# License: MIT

import os
from os import path
import openneuro
from mne.datasets import sample
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap

from naplib.io import load_bids

###############################################################################
# Download EEG data from OpenNeuro
# --------------------------------
# 
# Import data and get auditory spectrogram which will be used as stimulus.

dataset = 'ds002778'
subject = 'pd6'

bids_root = path.join(path.dirname(sample.data_path()), dataset)
print(bids_root)
if not path.isdir(bids_root):
    os.makedirs(bids_root)

openneuro.download(dataset=dataset, target_dir=bids_root,
                   include=[f'sub-{subject}'])


###############################################################################
# Read the data into a Data object
# --------------------------------

# We are only interested in the 32-channel EEG data as the responses, so select those channels
resp_channels = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7',
                 'P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2',
                 'C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']

data = load_bids(root=bids_root, subject=subject, datatype='eeg', task='rest', suffix='eeg', session='off', resp_channels=resp_channels)


###############################################################################
# Compute Alpha Theta Ratio
# -------------------------
# 
# Let's compute the Alpha/Theta Ratio in each channel. We use the log-value so that ratios above 1 are positive and ratios below 1 are negative, which makes the resulting topomap more clear.

def log_alpha_theta_ratio(response, sfreq):
    '''response should be of shape (time * channels)'''
    # must transpose response for mne function
    alpha_psd, _ = mne.time_frequency.psd_array_welch(response.T, sfreq, fmin=8, fmax=13, verbose=False) # psd is shape (channels * freqs)
    alpha_psd = alpha_psd.mean(-1)
    
    theta_psd, _ = mne.time_frequency.psd_array_welch(response.T, sfreq, fmin=4, fmax=8, verbose=False) # psd is shape (channels * freqs)
    theta_psd = theta_psd.mean(-1)
    
    return np.log(alpha_psd / theta_psd)
    

alpha_theta_ratio = [log_alpha_theta_ratio(trial['resp'], trial['sfreq']) for trial in data]


###############################################################################
# Visualize Results with MNE
# --------------------------
# 
# The Data contains the mne_info attribute (data.mne_info) which we can use for plotting
# This info is an instance of mne.Info, and it contains measurement information
# like channel names, locations, etc, as well as other metadata


# First, we need to set the montage (i.e. the arrangement of electrodes) so that the channels can be plotted properly
# Here, we set it to the standard 10-20 system, but many options are available if the data were recorded in a different
# montage. See https://mne.tools/dev/generated/mne.channels.make_standard_montage.html for details
data.mne_info.set_montage('standard_1020')

fig, ax = plt.subplots()
ax.set_title('Trial 1 Alpha/Theta Ratio')
plot_topomap(alpha_theta_ratio[0], data.mne_info, axes=ax)
plt.show()


