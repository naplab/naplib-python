import pytest
import numpy as np
import matplotlib.pyplot as plt

from naplib.visualization import shaded_error_plot

def test_errorplot_bad_fmt_type_error():

    _, ax = plt.subplots(1,1)

    with pytest.raises(TypeError):
        shaded_error_plot([0,1,2,3], np.random.rand(4,2), {'fmt': 'bad'}, ax=ax, err_method='stderr')

def test_errorplot_no_args_error():

    with pytest.raises(ValueError):
        shaded_error_plot()

def test_errorplot_too_many_args_error():

    _, ax = plt.subplots(1,1)

    with pytest.raises(ValueError):
        shaded_error_plot([0,1,2,3], np.random.rand(4,2), 'r--', 'bad_arg', ax=ax, err_method='stderr')

def test_errorplot_bad_err_method():

    _, ax = plt.subplots(1,1)

    with pytest.raises(ValueError) as err:
        shaded_error_plot([0,1,2,3], np.random.rand(4,2), ax=ax, err_method='not an option')
    assert 'is a string but is not one of' in str(err)

    with pytest.raises(ValueError) as err:
        shaded_error_plot([0,1,2,3], np.random.rand(4,2), ax=ax, err_method=-0.1)
    assert 'is a float then it must be in the range (0, 1]' in str(err)

    with pytest.raises(ValueError) as err:
        shaded_error_plot([0,1,2,3], np.random.rand(4,2), ax=ax, err_method=1.2)
    assert 'is a float then it must be in the range (0, 1]' in str(err)

    with pytest.raises(ValueError) as err:
        shaded_error_plot([0,1,2,3], np.random.rand(4,2), ax=ax, err_method={})
    assert ' must be either a string or a float' in str(err)

def test_errorplot_bad_reduction():

    _, ax = plt.subplots(1,1)

    with pytest.raises(ValueError) as err:
        shaded_error_plot([0,1,2,3], np.random.rand(4,2), ax=ax, reduction='bad')
    assert 'reduction must be either' in str(err)

def test_errorplot_error_region_median():
    x = np.array([-1,0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_mean = np.median(y, axis=1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, reduction='median', ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_median_propogate():
    x = np.array([-1,0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    y[2,0] = np.nan
    # y_std = y.std(1)
    y_mean = np.median(y, axis=1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, reduction='median', ax=ax, err_method='stderr', nan_policy='propogate')
    nan_mask = np.isnan(y_mean)
    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1][~nan_mask], y_mean[~nan_mask], atol=1e-10)

def test_errorplot_error_region_percentile_propogate():
    x = np.array([-1,0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    y[2,0] = np.nan
    # y_std = y.std(1)
    y_mean = y.mean(1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, ax=ax, err_method=0.95, nan_policy='propogate')
    nan_mask = np.isnan(y_mean)
    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1][~nan_mask], y_mean[~nan_mask], atol=1e-10)

def test_errorplot_error_region_percentile_omit():
    x = np.array([-1,0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    y[2,0] = np.nan
    # y_std = y.std(1)
    y_mean = np.nanmean(y, axis=1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, ax=ax, err_method=0.95, nan_policy='omit')
    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_stderr():
    x = np.array([-1,0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_mean = y.mean(1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_percentile_no_x_given():
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_mean = y.mean(1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(y, ax=ax, err_method=0.95)

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)


def test_errorplot_error_region_stderr_no_x_given():
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_mean = y.mean(1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(y, ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_stderr_no_x_given_with_fmt_string():
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_mean = y.mean(1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(y, 'r--*', ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_stderr_with_fmt_string():
    x = np.array([-1, 0,1,2,3])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = 9
    y[4,1] = -0.5
    # y_std = y.std(1)
    y_mean = y.mean(1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, 'r--*', ax=ax, err_method='stderr')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([-1,0,1,2,3]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_std_withnan_omit():
    x = np.array([0,1,2,3,4])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = np.nan
    y[4,1] = np.nan
    y_mean = np.nanmean(y,1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, ax=ax, err_method='std', nan_policy='omit')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    assert np.allclose(ax.lines[0].get_data()[1], y_mean, atol=1e-10)

def test_errorplot_error_region_std_withnan_propogate():
    x = np.array([0,1.,2,3,4])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = np.nan
    y[4,1] = np.nan
    y_mean = np.mean(y,1)

    _, ax = plt.subplots(1,1)
    shaded_error_plot(x, y, ax=ax, err_method='std', nan_policy='propogate')

    assert np.allclose(ax.lines[0].get_data()[0], np.array([0,1,2,3,4]), atol=1e-10)
    nan_mask = np.isnan(y_mean)
    assert np.allclose(ax.lines[0].get_data()[1][~nan_mask], y_mean[~nan_mask], atol=1e-10)
    assert np.isnan(ax.lines[0].get_data()[1][nan_mask]).all()

def test_errorplot_error_region_std_withnan_raise():
    x = np.array([0,1,2,3,4])
    y = np.arange(1, 31).reshape((5,6)).astype('float')
    y[0,2] = np.nan
    y[4,1] = np.nan

    _, ax = plt.subplots(1,1)
    with pytest.raises(ValueError) as err:
        shaded_error_plot(x, y, ax=ax, err_method='std', nan_policy='raise')
    assert 'Found nan in input' in str(err)

