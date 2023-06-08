"""
==========================
Array operations in naplib
==========================

How to easily process Data objects.
"""
# Author: Gavin Mischler
# 
# License: MIT


import numpy as np

import naplib as nl


data = nl.io.load_speech_task_data()
print(f'This Data contains {len(data)} trials')
print(f"Each trial has {data['resp'][0].shape[1]} channels.")


###############################################################################
# Quickly perform operations across trials
# ----------------------------------------
# 
# In many cases, we may want to perform some array operation on all of our data,
# such as performing PCA or TSNE to reduce the 30 channels down to 2 dimensions.
# However, we want to fit these models using all of the trials' data, and then
# take the 2-dimensional data and split it back into the same trial lengths.
# To quickly do this, we can use the ```concat_apply``` function in naplib,
# which concatenates data across trials, performs some function on it, and then
# returns the result as a list of trials once again.


from naplib.array_ops import concat_apply
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_resps = concat_apply(data['resp'], pca.fit_transform)

# Print the shapes of the first four trials before and after PCA
for n in range(4):
    print(f"Response shape: {data['resp'][n].shape}, PCA shape: {pca_resps[n].shape}")


###############################################################################
# Include a sliding window in the operations
# ------------------------------------------
# 
# Above we reduced 30 channels down to 2 at each time instant, but what if we
# want to consider a small time window of the responses, for example a
# 3-sample window of each electrode's response, and then reduce this 90 dimensions
# down to 2? To do this, we can combine the ```concat_apply``` with ```sliding_window```.



from naplib.array_ops import sliding_window

# The sliding_window function can be used to generate sliding windows of data
window_len = 3
window_key_idx = 1 # This produces a noncausal window centered around the given point. Note: window_key_idx=0 -> causal, window_key_idx=window_len-1 -> anticausal, 0<window_key_idx<window_len-1 -> noncausal
windowed_resps = [sliding_window(trial, window_len, window_key_idx=window_key_idx) for trial in data['resp']]

for n in range(4):
    print(f"Response shape: {data['resp'][n].shape}, windowed response shape: {windowed_resps[n].shape}")

# now we can use concat_apply on the windowed responses to TSNE the 3*30 dimensions down to 2

windowed_resps = [x.reshape(x.shape[0], -1) for x in windowed_resps] # put the 3*30 dimensions together
pca = PCA(n_components=2)
pca_resps = concat_apply(windowed_resps, pca.fit_transform)

# Print the shapes of the first four trials before and after PCA
for n in range(4):
    print(f"Windowed response shape: {windowed_resps[n].shape}, PCA shape: {pca_resps[n].shape}")

###############################################################################
# Using custom functions with concat_apply
# ----------------------------------------
# 
# The ```concat_apply``` function takes as an argument a Callable function
# which can be called on an array to produce another array. If we want to
# do more complicated things, we can define the callable function ourselves.
# For example, we may want to perform PCA on a window of each channel
# independently, reducing the window_len dimensions of data for a single
# channel down to 2 dimensions.


# first, get the windowed responses for each channel, which are of shape (time, window_len, channels) for each trial
window_len = 10 # use a window size of 10 samples (100 ms in our data)
window_key_idx = 0 # here we are using a causal window
windowed_resps = [sliding_window(trial, window_len, window_key_idx=window_key_idx) for trial in data['resp']]

print(f'First trial windowed response shape: {windowed_resps[0].shape}\n')

# Define our custom Callable
# It must take a np.ndarray of shape (time, ...) as input and produce a np.ndarray with the same first dimension shape.
# Behind the scenes, the concat_apply with concatenate all the trials across the first dimension, pass them to this
# function, and then split the result back into the separate trials
def single_channel_PCA(arr):
    # base on our windowing, the input arr will be of shape (time, 10, num_channels)
    output_shape = (arr.shape[0], 2, arr.shape[-1])
    results = np.empty(output_shape)
    for channel in range(arr.shape[-1]):
        pca = PCA(n_components=2)
        results[:,:,channel] = pca.fit_transform(arr[:,:,channel])
    return results

pca_resps_singlechannel = concat_apply(windowed_resps, single_channel_PCA)

# Print the shapes of the first four trials before and after PCA
for n in range(4):
    print(f"Windowed response shape: {windowed_resps[n].shape}, PCA shape: {pca_resps_singlechannel[n].shape}")

