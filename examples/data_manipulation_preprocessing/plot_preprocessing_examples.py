"""
==================================
Preprocessing Neural Response Data
==================================

Preprocessing iEEG and EEG data.

In this example, we demonstrate how to use various preprocessing methods to prepare neural data for analysis using naplib-python.


"""
# Author: Gavin Mischler
# 
# License: MIT

import numpy as np
import matplotlib.pyplot as plt

import naplib as nl
from naplib.preprocessing import filter_butter, normalize, phase_amplitude_extract, filter_line_noise
from naplib.stats import responsive_ttest

###############################################################################
# 1. Preprocessing intracranial EEG (iEEG) responses
# --------------------------------------------------
# 
# The first step is often to extract different frequency band envelopes, such as the highgamma envelope, from the raw data.

# Since we don't have raw iEEG responses to load, we generate random data to simulate the first step.
num_trials = 5
x = [np.random.rand(10000, 6) for _ in range(num_trials)] # 5 trials, each with 10000 samples and 6 channels
fs = [500 for _ in range(num_trials)]

data = nl.Data({'resp': x, 'dataf': fs})

# First, we want to remove the line noise in our data
# It doesn't really matter here since our data is randomly generated,
# but we should always do this in practice
line_noise_freq = 60

data['resp'] = filter_line_noise(data, f=line_noise_freq)


# Now, we extract alpha and highgamma frequency bands (both their phase and envelope/amplitude)
freq_bands = [[8, 13], [70, 150]]
bandnames = ['alpha', 'highgamma']

# the output is a Data object with a phase and amplitude field for each frequency band
phase_amp = phase_amplitude_extract(data, Wn=freq_bands, bandnames=bandnames)


# plot the phase and amplitude for the one second for a single electrode
t = np.arange(0, 10000/500, 1/500)
fig, axes = plt.subplots(1,2,figsize=(10,3))
axes[0].plot(t[500:1000], phase_amp[0]['alpha phase'][500:1000,0])
axes[1].plot(t[500:1000], phase_amp[0]['alpha amp'][500:1000,0])
axes[0].set_title('alpha phase')
axes[1].set_title('alpha envelope')
plt.show()

fig, axes = plt.subplots(1,2,figsize=(10,3))
axes[0].plot(t[500:1000], phase_amp[0]['highgamma phase'][500:1000,0])
axes[1].plot(t[500:1000], phase_amp[0]['highgamma amp'][500:1000,0])
axes[0].set_title('Highgamma phase')
axes[1].set_title('Highgamma envelope')
plt.show()

###############################################################################
# Process Highgamma Envelope Responses
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Here, we load a dataset that already contains highgamma envelope responses to clean speech.

data = nl.io.load_speech_task_data()
data.fields

###############################################################################
# Keep only speech-responsive electrodes
# 
# Using a t-test between responses to silence and speech (the onset of which is described by the 'befaft' field of the Data),
# we can determine which electrodes are speech-responsive. Since this simulated data has entirely
# responsive electrodes, we first set 2 of the electrodes to be random noise.

# this indicates that for this trial, the response includes 1 second of silence before the
# stimulus began, and 1 second at the end after the stimulus ended
print(f"befaft period: {data['befaft'][0]}")

rng = np.random.default_rng(1)
for trial in data:
    trial['resp'][:,3:5] = .25*rng.normal(size=(trial['resp'].shape[0],2))

plt.figure()
plt.plot(np.linspace(0, 4, 400), data['resp'][0][:400], alpha=0.6)
plt.vlines(data['befaft'][0][0], -0.75, 2.5, label='Stim Onset', color='r', linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.show()


# Perform t-test to remove electrodes that are not responsive
# This function performs FDR correction by default, but many parameters can be changed to alter
# the test
data_responsive, stats = responsive_ttest(data)
print(stats.keys()) # statistics computed


# now there are only 8 electrodes remaining
print(data_responsive[0].shape)
data['resp'] = data_responsive # set the responses to only be these good electrodes


# plot some of the statistics computed
# the p-values computed by response_ttest are corrected for multiple comparisons already

fig, axes = plt.subplots(2,1,sharex=True)
axes[0].plot(np.abs(stats['stat']), label='T-value')
axes[0].legend()
axes[1].plot(np.abs(stats['pval']), 'o-', label='pval')
axes[1].hlines(stats['alpha'], 0, 10, color='r', label='alpha-level')
axes[1].legend()
axes[1].set_xlabel('Electrode')
plt.tight_layout()
plt.show()


###############################################################################
# 2. Frequency bandpass filtering (more used for EEG than iEEG)
# -------------------------------------------------------------
# 
# Let's imagine our data is EEG data instead of iEEG, and we want to extract different
# frequency bands for further analysis
# 
# * delta (0.5-4Hz)
# * theta (4-8 Hz)
# * alpha (8-13 Hz)
# * beta (14-30Hz)

# filter the responses to different bands and set them as new fields of the Data
data['delta_resp'] = filter_butter(data, field='resp', Wn=[0.5, 4], order=5)
data['theta_resp'] = filter_butter(data, field='resp', Wn=[4, 8], order=5)
data['alpha_resp'] = filter_butter(data, field='resp', Wn=[8, 13], order=5)
data['beta_resp'] = filter_butter(data, field='resp', Wn=[14, 30], order=5)


# To make sure the filter bands are correct or meet our requirements, get the filters used and plot them
# If we are not satisfied with the filters, we can set the ``order`` parameter in the ``filter_butter`` function to increase the filter order.
theta_resp, filters = filter_butter(data, Wn=[4, 8], return_filters=True, order=5)

# plot frequency response
fig, ax = plt.subplots()
nl.visualization.freq_response(filters[0], fs=data[0]['dataf'], ax=ax)
plt.show()

# use a higher order to get steeper cutoff region
theta_resp_2, filters_2 = filter_butter(data, Wn=[4, 8], order=10, return_filters=True)

# plot frequency response
fig, ax = plt.subplots()
nl.visualization.freq_response(filters_2[0], fs=data[0]['dataf'], ax=ax)
plt.show()


###############################################################################
# Normalize filtered responses and plot
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Now that we have bandpassed responses, normalize each one (by z-scoring).

data['delta_resp'] = normalize(data, field='delta_resp')
data['theta_resp'] = normalize(data, field='theta_resp')
data['alpha_resp'] = normalize(data, field='alpha_resp')
data['beta_resp'] = normalize(data, field='beta_resp')

trial = 0
t = np.arange(0, 10*100) / data['dataf'][trial]

fig, axes = plt.subplots(4,1,figsize=(12,8), sharex=True)

axes[0].plot(t, data['delta_resp'][trial][:10*100])
axes[0].set_title('Delta')
axes[1].plot(t, data['theta_resp'][trial][:10*100])
axes[1].set_title('Theta')
axes[2].plot(t, data['alpha_resp'][trial][:10*100])
axes[2].set_title('Alpha')
axes[3].plot(t, data['beta_resp'][trial][:10*100])
axes[3].set_title('Beta')
axes[3].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
