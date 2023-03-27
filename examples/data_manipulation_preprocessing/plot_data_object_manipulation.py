"""
======================
Data Objects in naplib
======================

Basic operations on the Data object.

This example shows some of the basic operations that can be performed on the Data object.
We first import data, then process it.
"""
# Author: Gavin Mischler
# 
# License: MIT


import numpy as np
import matplotlib.pyplot as plt

import naplib as nl

###############################################################################
# Creating a Data Object from Scratch
# -----------------------------------
# 
# If we have data in our own custom format or as numpy arrays, we can easily
# put them into a Data object. Here, we generate response data randomly,
# and the stimuli are pure tone sine waves.


nCh = 20 # number of recording channels in the responses
fs = 100 # sampling rate of the response data
audio_fs = 16000 # sampling rate of the audio stimuli
trial_lengths = [5.0, 6.7, 5.8] # length of each trial (in seconds)
tone_frequencies = [2, 5, 8] # frequency (Hz) of each trial's stimulus tone

trial_responses = [np.random.normal(size=(int(L*fs), nCh)) for L in trial_lengths]
trial_stimuli = [np.sin(2 * np.pi * freq * np.linspace(0, L, int(L*audio_fs))) for freq, L in zip(tone_frequencies, trial_lengths)]

data_dict = {'name': ['trial-1','trial-2','trial-3'],
             'soundf': [audio_fs, audio_fs, audio_fs],
             'sound': trial_stimuli,
             'dataf': [fs, fs, fs],
             'resp': trial_responses
             }

data = nl.Data(data_dict)
print(data)

###############################################################################
# We can plot the stimuli or responses easily by first using integer indexing
# to select which trial, and then field-name indexing to select a field

fig, axes = plt.subplots(2,1, figsize=(8,3))
axes[0].set_title('trial-1')
axes[0].plot(data[0]['sound'])
axes[1].plot(data[0]['resp'])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,1, figsize=(8,3))
axes[0].set_title('trial-2')
axes[0].plot(data[1]['sound'])
axes[1].plot(data[1]['resp'])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2,1, figsize=(8,3))
axes[0].set_title('trial-2')
axes[0].plot(data[2]['sound'])
axes[1].plot(data[2]['resp'])
plt.tight_layout()
plt.show()


###############################################################################
# Processing Pre-existing Data
# ----------------------------
# 
# Here, we load a sample Data object that we can analyze further

# We can import a sample data object to look at
data = nl.io.load_speech_task_data()

# Let's look at the data we have
print(len(data)) # number of trials/stimuli recorded
print(data.fields) # data fields available for each trial


###############################################################################
# We can index into the trials using integer indices, or index into the fields with a string

third_trial = data[2]
print(third_trial)
name_field = data['name']
print(name_field)

###############################################################################
# We can also extract a smaller Data object which contains only a set of the
# fields by indexing a list of fieldnames

smaller_data = data[['name','soundf','resp']]
print(smaller_data)


###############################################################################
# We can also extract a smaller Data by indexing with slicing
# or a list of trial indices
smaller_data2 = data[[0,2,4,9]]
print(smaller_data2)
smaller_data3 = data[1:5]
print(smaller_data3)


###############################################################################
# We can also iterate over the trials and get interesting information

for t, trial in enumerate(data[:2]): # only look at the first 2 trials
    print(f'trial {t}-----')
    print(type(trial)) # each trial is a dictionary object
    print(trial['name']) # we can access a single field of this trial, like the name


###############################################################################
# We can also access a single field of all the trials by using data.get_field(), or just brackets

print([resp.shape for resp in data.get_field('resp')]) # shape of each trial: (time * num_elecs)
print([resp.shape for resp in data['resp']]) # equivalent to above



###############################################################################
# We can also join different subjects data, such as concatenating their "resp"
# fields together to get the responses to all subjects' electrodes

import copy
data_other = copy.deepcopy(data)
print("this subject has {} electrodes...".format(data_other['resp'][0].shape[1])) # dimensions of 


# join the "resp" fields, now we have all electrodes data together
joined_responses = nl.join_fields([data, data_other], fieldname='resp', axis=-1)
print('after concatenating, each trial has shape:')
print([resp.shape for resp in joined_responses])


###############################################################################
# Adding a new field to the Data
# ------------------------------
# 
# If we have some more data, labels, or something that we want to store in our
# Data as well, we can add a field to the Data. To add a field, you must pass
# in either a list of the same length as the Data which contains the data for
# this field for each trial, or a multidimensional array where the first index
# is the length of the Data.


# for example, let's add a new name for each and call the field "test_name"
new_field_data = [f'test{i}' for i in range(len(data))]
data['test_name'] = new_field_data
print(data.fields) # now we have a new field
print(data[0]['test_name']) # let's print the data in this field for the first trial


###############################################################################
# Processing a Data Object
# ------------------------
# 
# Many functions in naplib operate directly on data objects


# now let's zscore the response data over time (and across all trials)
from naplib.preprocessing import normalize

# pass in our data to be z-scored
norm_resp = normalize(data=data, field='resp', axis=0, method='zscore')

# we can also directly pass in the field as the data
norm_resp2 = normalize(field=data['resp'], axis=0, method='zscore')

# print out the standard deviation before and after normalization
# Note: the standard dev. is not exactly 1 because it was computed over the full out struct,
# not for each trial individually

print('old data standard deviation: ')
print(data[0]['resp'].std(axis=0))
print('new data standard deviation (method 1): ')
print(norm_resp[0].std(axis=0))
print('new data standard deviation (method 2): ')
print(norm_resp2[0].std(axis=0))


# looks good, so let's replace the 'resp' field in our Data with this new normalized resp
data['resp'] = norm_resp
print('new data standard deviation after replacing: ')
print(data[0]['resp'].std(axis=0)) # now the normalized data is in the 'resp' field

