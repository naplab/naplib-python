import pytest
import numpy as np
import matplotlib.pyplot as plt

from naplib.visualization import kde_plot

def test_kde_single_line():

    expected_line1_x = np.array([-2.75921053, -2.72823507, -2.69725962, -2.66628416])
    rng = np.random.default_rng(1)
    data = rng.normal(size=(100,))
    
    _, ax = plt.subplots(1,1)
    kde_plot(data[:50], bw_method=0.25, bins=50)
    assert np.allclose(ax.lines[0].get_data()[0][20:24], expected_line1_x)
    plt.close()

    _, ax = plt.subplots(1,1)
    kde_plot([xx for xx in data[:50]], bw_method=0.25, bins=50)
    assert np.allclose(ax.lines[0].get_data()[0][20:24], expected_line1_x)
    plt.close()

def test_plot_densities_individually_on_same_axes():

    expected_line1_x = np.array([-2.75921053, -2.72823507, -2.69725962, -2.66628416])
    expected_line2_y = np.array([0.03010354, 0.03247873, 0.03460289, 0.03641259])

    rng = np.random.default_rng(1)
    data = rng.normal(size=(100,))
    data[50:] += 0.5 # shift the second half of the samples
    groupings = np.array(['G0'] * 100) # define grouping vector
    groupings[50:] = 'G1' # set a different label for the samples we shifted

    _, ax = plt.subplots(1,1)
    kde_plot(data[:50], ax=ax, bw_method=0.25, bins=10)
    kde_plot(data[50:], ax=ax, bw_method=0.25, bins=13)
    assert np.allclose(ax.lines[0].get_data()[0][20:24], expected_line1_x)
    assert np.allclose(ax.lines[1].get_data()[1][20:24], expected_line2_y)
    plt.close()

def test_kde_line_same_even_if_change_data_format_but_bw_method_constant():

    expected_line1_x = np.array([-2.75921053, -2.72823507, -2.69725962, -2.66628416])
    expected_line2_y = np.array([0.03010354, 0.03247873, 0.03460289, 0.03641259])

    rng = np.random.default_rng(1)
    data = rng.normal(size=(100,))
    data[50:] += 0.5 # shift the second half of the samples
    groupings = np.array(['G0'] * 100) # define grouping vector
    groupings[50:] = 'G1' # set a different label for the samples we shifted

    _, ax = plt.subplots(1,1)
    ax = kde_plot(data, groupings=groupings, bw_method=0.25, bins=10, color=['k','r'])
    assert np.allclose(ax.lines[0].get_data()[0][20:24], expected_line1_x)
    assert np.allclose(ax.lines[1].get_data()[1][20:24], expected_line2_y)
    plt.close()

    _, ax = plt.subplots(1,1)
    data_list = [data[:50],data[50:]]
    ax = kde_plot(data_list, groupings=['One','Two'], bw_method=0.25, bins=7)
    assert np.allclose(ax.lines[0].get_data()[0][20:24], expected_line1_x)
    assert np.allclose(ax.lines[1].get_data()[1][20:24], expected_line2_y)
    plt.close()

    _, ax = plt.subplots(1,1)
    data_mat = np.concatenate([data[:50,np.newaxis],data[50:,np.newaxis]], axis=1)
    ax = kde_plot(data_mat, bw_method=.25, bins=20, color=['k','r'])
    assert np.allclose(ax.lines[0].get_data()[0][20:24], expected_line1_x)
    assert np.allclose(ax.lines[1].get_data()[1][20:24], expected_line2_y)
    plt.close()


def test_legend_correct_for_different_groupings():

    rng = np.random.default_rng(1)
    data = rng.normal(size=(100,))
    data[50:] += 0.5 # shift the second half of the samples
    groupings = np.array(['G0'] * 100) # define grouping vector
    groupings[50:] = 'G1' # set a different label for the samples we shifted

    _, ax = plt.subplots(1,1)
    ax = kde_plot(data, groupings=groupings, bw_method=0.25, bins=10, color=['k','r'])
    assert ax.lines[0].get_label() == 'G0'
    plt.close()

    _, ax = plt.subplots(1,1)
    ax = kde_plot(data, groupings=groupings=='G0', bw_method=0.25, bins=10, color=['k','r'])
    assert ax.lines[1].get_label() == 'True'
    plt.close()

    _, ax = plt.subplots(1,1)
    data_mat = np.concatenate([data[:50,np.newaxis],data[50:,np.newaxis]], axis=1)
    ax = kde_plot(data_mat, groupings=['One','Two'], bw_method=0.25, bins=10, color=['k','r'])
    assert ax.lines[0].get_label() == 'One'
    plt.close()

def test_plot_on_existing_axes():

    rng = np.random.default_rng(1)
    data = rng.normal(size=(100,))
    data[50:] += 0.5 # shift the second half of the samples
    groupings = np.array(['G0'] * 100) # define grouping vector
    groupings[50:] = 'G1' # set a different label for the samples we shifted

    _, ax = plt.subplots(1,1)
    ax = kde_plot(data, groupings=groupings, ax=ax, bw_method=0.25, bins=10)
    assert ax.get_xlabel() == ''
    plt.close()

    _, ax = plt.subplots(1,1)
    ax.set_xlabel('New x-label')
    ax.set_ylabel('New y-label')
    ax = kde_plot(data, groupings=groupings, ax=ax, bw_method=0.25, bins=10)
    assert ax.get_xlabel() == 'New x-label'
    assert ax.get_ylabel() == 'New y-label'
    plt.close()

def test_bad_groupings_length():
    # 1D case
    with pytest.raises(ValueError) as err:
        kde_plot(np.arange(10), groupings=[1,2,1,1,2])
    assert 'data and groupings must be same length' in str(err)

    # 2D array case
    with pytest.raises(TypeError) as err:
        kde_plot(np.random.rand(10,2), groupings=[1,2,1,1,2])
    assert 'Invalid format for groupings when data is multidimensional numpy' in str(err)

    # list of 1D array case
    with pytest.raises(ValueError) as err:
        kde_plot([np.arange(10), np.arange(1,8)], groupings=[1,2,1,1,2])
    assert 'groupings must be same length as data if data is given as list' in str(err)

def test_bad_data_type():
    with pytest.raises(TypeError) as err:
        kde_plot({'d': [1,2,3]})
    assert 'data must be either a np.ndarray' in str(err)

    with pytest.raises(TypeError) as err:
        kde_plot([np.arange(5), 1, 2])
    assert 'If data is a list, each element must be a nump' in str(err)

def test_bad_number_of_colors():
    with pytest.raises(ValueError) as err:
        kde_plot(np.array([1,1,2,3]), color=['k','r'])
    assert 'number of colors provided must match number of groups' in str(err)

