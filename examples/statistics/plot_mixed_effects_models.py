"""
===========================
Linear Mixed Effects Models
===========================

Analyzing linear mixed effects models.

In this tutorial, we will demonstrate the use of the linear mixed effects model to identify fixed effects. These models are useful when data has some non-independence. For example, if half of the samples of the data come from subject A, and the other half come from subject B, but we want to remove the effect of subject identify and look at only the impact of the features that interest us.
"""
# Author: Gavin Mischler
# 
# License: MIT


import numpy as np
import matplotlib.pyplot as plt

import naplib as nl


###############################################################################
# Generate Non-independent Data
# -----------------------------
# 
# First, we will simulate some non-independent data where subject identity causes a
# random effect on the output variable, which is also dependent on two independent variables.

# The true effects of each feature will be 1 for feature 1 and -2 for feature 2
betas = np.array([1, -2])

def generate_data(N_per_subject=50, n_subjects=5):
    '''
    Generates independent variables and dependent variables where there is
    a random effect of subject identity.
    There are n_subjects, and each has N_per_subject data points.
    '''
    X = [] # independent variables
    Y = [] # dependent variable
    subject_ID = [] # subject ID random effect variable
    data_mean = np.array([0., 0.]).reshape(1,-1)
    
    subj_betas = np.array([1, 0])
    for i in range(n_subjects):
        X_thissubject = np.random.uniform(size=(N_per_subject,2)) + data_mean
        Y_thissubject = X_thissubject @ betas + data_mean @ subj_betas - 3*data_mean[0,0] + np.random.normal(scale=0.25, size=(N_per_subject,))
        X.append(X_thissubject)
        Y.append(Y_thissubject)
        subject_ID.append(i*np.ones_like(Y_thissubject))
        data_mean += np.array([1, 0]).reshape(1,-1)
        
    fig, axes = plt.subplots(1,2,figsize=(12,6), sharey=True)
    for i, (x, y) in enumerate(zip(X, Y)):
        axes[0].scatter(x[:,0], y, label=f'Subj {i}')
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Output variable')
        axes[1].scatter(x[:,1], y, label=f'Subj {i}')
        axes[1].set_xlabel('Feature 2')
    plt.legend()
    plt.show()
    
    return np.concatenate(X), np.concatenate(Y), np.concatenate(subject_ID)
    
X, Y, subject_ID = generate_data()
print((X.shape, Y.shape, subject_ID.shape))


###############################################################################
# Use mixed effects model to identify fixed effects of features 1 and 2
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

###############################################################################
# Examine the model which uses the random effect of subject ID

# this model uses the random effect of subject ID
model = nl.stats.LinearMixedEffectsModel()
varnames = ['Feature 1', 'Feature 2', 'Subject', 'Output Variable']
model.fit(X, Y, random_effect=subject_ID, varnames=varnames)

print(model.summary())

###############################################################################
# Examine the model which does not use the random effect of subject ID

# this model does not use the random effect of subject ID
model2 = nl.stats.LinearMixedEffectsModel()
varnames = ['Feature 1', 'Feature 2', 'Output Variable']
model2.fit(X, Y, varnames=varnames)

model2.summary()

###############################################################################
# Visualize the summaries using effect plots
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# As we can see in the Coef. tab of the summaries above, or the fixed effect plots below, the first model, which used the random effect of subject, recovered the true beta weights for each feature. On the other hand, the second model found the wrong weight for feature 1, since the subject identity had a large effect on the first feature. Its estimate of the fixed effect of the second feature was close to the truth, but still wrong and with a larger confidence interval.

plt.figure()
model.plot_effects()
plt.show()

plt.figure()
model2.plot_effects()
plt.show()

