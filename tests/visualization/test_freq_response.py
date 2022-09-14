import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter

from naplib.visualization import freq_response

def test_plot_Hz():
    b, a = butter(4, 10, 'low', fs=100)
    fig, ax = plt.subplots(1,1)
    freq_response((b,a), fs=100, units='Hz')

def test_plot_rad():
    b, a = butter(4, 10, 'low', fs=100)
    fig, ax = plt.subplots(1,1)
    freq_response((b,a), fs=100, units='rad/s')
