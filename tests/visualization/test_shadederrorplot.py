import pytest
import numpy as np
import matplotlib.pyplot as plt

from naplib.visualization import shadederrorplot

def test_errorplot_bad_fmt_type_error():

    fig, ax = plt.subplots(1,1)

    with pytest.raises(TypeError):
        shadederrorplot([0,1,2,3], np.random.rand(4,2), {'fmt': 'bad'}, ax=ax, err_method='stderr')

def test_errorplot_no_args_error():

    with pytest.raises(ValueError):
        shadederrorplot()

def test_errorplot_too_many_args_error():

    fig, ax = plt.subplots(1,1)

    with pytest.raises(ValueError):
        shadederrorplot([0,1,2,3], np.random.rand(4,2), 'r--', 'bad_arg', ax=ax, err_method='stderr')


def test_errorplot_error_region_stderr():
    x = np.array([-1,0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_err = np.nanstd(y, axis=1) / np.sqrt(y.shape[1])
    y_mean = y.mean(1)
    y_region = [y_mean - y_err, y_mean+y_err]

    fig, ax = plt.subplots(1,1)
    shadederrorplot(x, y, ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_stderr_no_x_given():
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_err = np.nanstd(y, axis=1) / np.sqrt(y.shape[1])
    y_mean = y.mean(1)
    y_region = [y_mean - y_err, y_mean+y_err]

    fig, ax = plt.subplots(1,1)
    shadederrorplot(y, ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_stderr_no_x_given_with_fmt_string():
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_err = np.nanstd(y, axis=1) / np.sqrt(y.shape[1])
    y_mean = y.mean(1)
    y_region = [y_mean - y_err, y_mean+y_err]

    fig, ax = plt.subplots(1,1)
    shadederrorplot(y, 'r--*', ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_stderr_with_fmt_string():
    x = np.array([-1, 0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_err = np.nanstd(y, axis=1) / np.sqrt(y.shape[1])
    y_mean = y.mean(1)
    y_region = [y_mean - y_err, y_mean+y_err]

    fig, ax = plt.subplots(1,1)
    shadederrorplot(x, y, 'r--*', ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_std_withnan_omit():
    x = np.array([0,1,2,3,4])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = np.nan
    y[4,1] = np.nan
    y_mean = np.nanmean(y,1)

    fig, ax = plt.subplots(1,1)
    shadederrorplot(x, y, ax=ax, err_method='std', nan_policy='omit')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_std_withnan_propogate():
    x = np.array([0,1.,2,3,4])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = np.nan
    y[4,1] = np.nan
    y_mean = np.mean(y,1)

    fig, ax = plt.subplots(1,1)
    shadederrorplot(x, y, ax=ax, err_method='std', nan_policy='propogate')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    nan_mask = np.isnan(y_mean)
    assert np.allclose(ax.lines[0].get_data()[1][~nan_mask], y_mean[~nan_mask], atol=1e-10)
    assert np.isnan(ax.lines[0].get_data()[1][nan_mask]).all()

def test_errorplot_error_region_std_withnan_raise():
    x = np.array([0,1,2,3,4])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = np.nan
    y[4,1] = np.nan
    y_mean = np.nanmean(y,1)

    fig, ax = plt.subplots(1,1)
    with pytest.raises(ValueError):
        shadederrorplot(x, y, ax=ax, err_method='std', nan_policy='raise')
