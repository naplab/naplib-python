import numpy as np
import matplotlib.pyplot as plt

def shadederrorplot(x, y, ax=None, plt_args={}, shade_args={}):
    '''
    Inputs
    ------
    x : shape (time,)
    y : shape (time, n_samples) 
    '''
    if ax is None:
        ax = plt.gca()
    y_mean = y.mean(1)
    y_err = y.std(1) / np.sqrt(y.shape[1])
    ax.plot(x, y_mean, **plt_args)
    ax.fill_between(x, y_mean-y_err, y_mean+y_err, **shade_args)