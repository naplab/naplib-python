"""
========================
Importing Data from BIDS
========================

Importing data from BIDS (Brain Imaging Data Structure)


Although there is no universal data storage format for neural data, the BIDS format has seen
widespread adoption. This tutorial demonstrates how to easily load data from a BIDS format
into a ``Data`` object using the `read_bids` function. Events stored in the BIDS format are
naturally segmented to form the "trials" in the ``Data``, facilitating easy data analysis.

**Note** 

Loading data from BIDS requires the additional dependency MNE-BIDS, which is not a hard dependency of `naplib-python`, so it must be installed separately. You can find installation instructions for `MNE-BIDS` [here](https://mne.tools/mne-bids/stable/install.html). This tutorial was adapted from a [similar tutorial](https://mne.tools/mne-bids/stable/auto_examples/read_bids_datasets.html#sphx-glr-auto-examples-read-bids-datasets-py) by `MNE-BIDS`.

Also, this tutorial makes use of an openneuro dataset for illustration purposes. The data downloaded here is about 1.5 GB from one subject.
"""
# Author: Gavin Mischler
# 
# License: MIT


import os
from os import path
import openneuro
from mne.datasets import sample
from mne_bids import print_dir_tree
import numpy as np
import matplotlib.pyplot as plt

from naplib.io import read_bids
from naplib.preprocessing import normalize

###############################################################################
# Download Dataset
# ----------------

dataset = 'ds002778'
subject = 'pd6'

bids_root = path.join(path.dirname(sample.data_path()), dataset)
print(bids_root)
if not path.isdir(bids_root):
    os.makedirs(bids_root)


openneuro.download(dataset=dataset, target_dir=bids_root,
                   include=[f'sub-{subject}'])

###############################################################################
# Look at the format of the BIDS file structure
# ---------------------------------------------


print_dir_tree(bids_root, max_depth=4)


###############################################################################
# Read file structure into a Data Object
# --------------------------------------
# 
# In this task, the stimulus recorded is simply a sudden spike in the 'stim' channels, so the associated events have duration=0. For the sake of this tutorial, we will cut the data into trials based on the 'onset' of one event and ending at the onset of the next. If the durations of the stimulus events were meaningful, we could cut by the 'duration' of each event instead. See the `read_bids` documentation for more details.

resp_channels = ['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7',
                 'P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2',
                 'C4','T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']

data = read_bids(root=bids_root, subject=subject, datatype='eeg', task='rest', suffix='eeg', session='off', resp_channels=resp_channels)


###############################################################################
# Visualize the Data
# ------------------
# 
# Visualize the stims and responses for the first 3 seconds of one of the events

sfreq = int(data[0]['sfreq']) # sampling frequency of the data

# normalize the responses for better visualization
data['resp'] = normalize(data, field='resp')

t = np.linspace(0, 3, 3*sfreq)

fig, axes = plt.subplots(2,1, figsize=(10,6))
axes[0].plot(t, data[1]['stim'][:3*sfreq,:])
axes[0].set_title('Event 1 Stim')

axes[1].plot(t, data[1]['resp'][:3*sfreq,:])
axes[1].set_title('Event 1 Response')
axes[1].set_label('Time (s)')
plt.tight_layout()
plt.show()

# All the metadata has been stored in the mne_info attribute of the Data
print(data.mne_info)


###############################################################################
# Storing a small portion of data before the onset of the stimulus event
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# We can use the `befaft` parameter to store an additional bit of the recording before and/or after each stimulus event region.

befaft = [1,0] # keep 1 second before the stimulus, nothing extra after

data2 = read_bids(root=bids_root, subject=subject, datatype='eeg', task='rest', suffix='eeg', session='off', befaft=befaft, resp_channels=resp_channels)

# Now, when we visualize we have a bit of data from before the stimulus onset

t = np.linspace(-1, 2, 3*sfreq)

data2['resp'] = normalize(data2, field='resp')

fig, axes = plt.subplots(2,1, figsize=(10,6))
axes[0].plot(t, data2[1]['stim'][:3*sfreq,:])
axes[0].set_title('Event 1 Stim')

axes[1].plot(t, data2[1]['resp'][:3*sfreq,:])
axes[1].set_title('Event 1 Response')
axes[1].set_label('Time (s)')
plt.tight_layout()
plt.show()

