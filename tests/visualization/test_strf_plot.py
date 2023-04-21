import pytest
import numpy as np
import matplotlib.pyplot as plt

from naplib.visualization import strf_plot

@pytest.fixture(scope='function')
def plot_fn():
    def _plot(coef, tmin=None, tmax=None, freqs=None, ax=None, smooth=True, return_ax=False):
        if return_ax:
            ax = strf_plot(coef, tmin=tmin, tmax=tmax, freqs=freqs, ax=ax, smooth=smooth, return_ax=return_ax)
        else:
            strf_plot(coef, tmin=tmin, tmax=tmax, freqs=freqs, ax=ax, smooth=smooth)
        plt.show()
        plt.close('all')
        return ax
    return _plot


def test_strf_plot_on_ax_no_smooth(plot_fn, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    fig, ax = plt.subplots(1,1)
    coef = np.random.rand(5,7)-0.5
    ax = strf_plot(coef, ax=ax, smooth=False)


def test_strf_plot_tmin_tmax(plot_fn, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    fig, ax = plt.subplots(1,1)
    coef = np.random.rand(5,7)-0.5
    strf_plot(coef, ax=ax, tmin=0, tmax=0.6)

def test_strf_plot_freqs(plot_fn, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    fig, ax = plt.subplots(1,1)
    coef = np.random.rand(5,7)-0.5
    strf_plot(coef, ax=ax, tmin=0, tmax=0.6, freqs=[100, 200, 300, 400, 500])
