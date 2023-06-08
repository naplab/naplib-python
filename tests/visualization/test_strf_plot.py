import numpy as np
import matplotlib.pyplot as plt

from naplib.visualization import strf_plot

def test_strf_plot_on_ax_no_smooth(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    _, ax = plt.subplots(1,1)
    coef = np.random.rand(5,7)-0.5
    ax = strf_plot(coef, ax=ax, smooth=False)

def test_strf_plot_tmin_tmax(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    _, ax = plt.subplots(1,1)
    coef = np.random.rand(5,7)-0.5
    strf_plot(coef, ax=ax, tmin=0, tmax=0.6)

def test_strf_plot_freqs(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    _, ax = plt.subplots(1,1)
    coef = np.random.rand(5,7)-0.5
    strf_plot(coef, ax=ax, tmin=0, tmax=0.6, freqs=[100, 200, 300, 400, 500])
