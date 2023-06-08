"""
==================
STRF Models Basics
==================

Basic STRF fitting tutorial.

This tutorial shows some of the basics of fitting STRFs and getting their predictions
for neural data in a Data object.

"""
# Author: Gavin Mischler
# 
# License: MIT


import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.linear_model import Ridge, ElasticNet

import naplib as nl
from naplib.visualization import strf_plot


###############################################################################
# Set up the data
# ---------------
# 
# Import data and get auditory spectrogram which will be used as stimulus.

data = nl.io.load_speech_task_data()

# This data contains the fields 'aud' and 'resp', which give the stimulus and neural responses
print(f"aud stimulus shape for first trial : {data[0]['aud'].shape}")
print(f"response shape for first trial : {data[0]['resp'].shape}")

# first, we normalize the responses
data['resp'] = nl.preprocessing.normalize(data=data, field='resp')

# get auditory spectrogram for each stimulus sound
data['spec'] = [nl.features.auditory_spectrogram(trial['sound'], 11025) for trial in data]

# make sure the spectrogram is the exact same size as the responses
data['spec'] = [resample(trial['spec'], trial['resp'].shape[0]) for trial in data] 

# Since the spectrogram is 128-channels, which is very large, we downsample it
print(f"before resampling: {data['spec'][0].shape}")

resample_kwargs = {'num': 32, 'axis': 1}
data['spec_32'] = nl.array_ops.concat_apply(data['spec'], resample, function_kwargs=resample_kwargs)

print(f"after resampling:  {data['spec_32'][0].shape}")


###############################################################################
# Fit Basic STRF Models
# ---------------------
# 
# Fit STRF models that have a receptive field of 300 ms in the past. We use
# normal Ridge Regression here. We could also use RidgeCV (which is the default)
# to automatically perform cross-validation on the penalty.


tmin = 0 # receptive field begins at time=0
tmax = 0.3 # receptive field ends at a lag of 0.4 seconds
sfreq = 100 # sampling frequency of data

# setting show_progress=False would disable the progress bar
strf_model = nl.encoding.TRF(tmin, tmax, sfreq, estimator=Ridge(10), show_progress=True)

# leave out 1 trial for testing
data_train = data[:-1]
data_test = data[-1:]

strf_model.fit(data=data_train, X='spec_32', y='resp')


###############################################################################
# Fit STRF Models with ElasticNet
# -------------------------------
# 
# Fit STRF models using ElasticNet (L1 and L2 penalty) regression.


# define the estimator to be used in this TRF model
estimator = ElasticNet(l1_ratio=0.01)

strf_model_2 = nl.encoding.TRF(tmin, tmax, sfreq, estimator=estimator)
strf_model_2.fit(data=data_train, X='spec_32', y='resp')


###############################################################################
# Analyze the STRFs Weights
# -------------------------
# 
# Visualize the STRF weights.

# compute model scores
scores = strf_model.score(data=data_test, X='spec_32', y='resp')
scores_2 = strf_model_2.score(data=data_test, X='spec_32', y='resp')

# we can access the STRF weights through the .coef_ attribute of the model
coef_ridge = strf_model.coef_
coef_elastic = strf_model_2.coef_

print(f'STRF shape (num_outputs, frequency, lag) = {coef_ridge.shape}')

# Now, visualize the STRF weights for the last electrode and for each model

freqs = [171, 5000]

elec = 9
model_1_coef, score_model_1 = coef_ridge[elec], scores[elec]
model_2_coef, score_model_2 = coef_elastic[elec], scores_2[elec]

fig, axes = plt.subplots(1,2,figsize=(6,2.5))
strf_plot(model_1_coef, tmin=tmin, tmax=tmax, freqs=freqs, ax=axes[0])
axes[0].set_title('Ridge, corr={:2.3f}'.format(score_model_1))
strf_plot(model_2_coef, tmin=tmin, tmax=tmax, freqs=freqs, ax=axes[1])
axes[1].set_title('ElasticNet, corr={:2.3f}'.format(score_model_2))
fig.suptitle(f'Electrode {elec}')
fig.tight_layout()
plt.show()
    

###############################################################################
# Analyze the STRF Scores
# -----------------------
# 
# Plot STRF prediction scores

# compare the scores from each model's predictions

fig, axes = plt.subplots(1,3,figsize=(10,3), sharex=True)
axes[0].hist(scores, 8)
axes[0].set_xlabel('Prediction correlation')
axes[0].set_title('Ridge Model Scores')
axes[1].hist(scores_2, 8)
axes[1].set_xlabel('Prediction correlation')
axes[1].set_title('ElasticNet Model Scores')
axes[2].scatter(scores, scores_2)
axes[2].set_xlabel('Ridge model correlation')
axes[2].set_ylabel('ElasticNet model correlation')
axes[2].plot([0,1],[0,1],'r--')
plt.tight_layout()
plt.show()


###############################################################################
# Analyze the STRFs Predictions
# -----------------------------
# 
# Compute and plot STRF predictions.

# We see that the two STRF models have nearly identical predictions for most electrodes

predictions = strf_model.predict(data=data_test, X='spec_32') # this is a list of the same length as data_test
predictions_2 = strf_model_2.predict(data=data_test, X='spec_32')

# plot the predictions for the first 10 seconds of the final trial for the last electrode
elec = 9

plt.figure(figsize=(12,3))
plt.plot(data_test['resp'][-1][:1000,elec], label='neural')
plt.plot(predictions[-1][:1000,elec], label='Ridge pred, corr={:2.2f}'.format(scores[elec]))
plt.plot(predictions_2[-1][:1000,elec], label=f'ElasticNet pred, corr={scores_2[elec]:2.2f}')
plt.xticks([0, 500, 1000], ['0', '5', '10'])
plt.xlabel('Time (s)')
plt.legend()
plt.show()

