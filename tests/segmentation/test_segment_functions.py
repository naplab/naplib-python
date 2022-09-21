import pytest
import numpy as np

from naplib.segmentation import get_label_change_points
from naplib.segmentation import segment_around_label_transitions
from naplib.segmentation import electrode_lags_fratio
from naplib import Data

@pytest.fixture(scope='module')
def outstruct():
    labels1 = [np.array([0, 0, 1, 1, 1, 3, 3, 3]), np.array([-1, 2, 2, 0, 0, 0])]
    labels2 = [np.array([1, 2, 3, 4, 5, 6, 7, 8]), np.array([8, 7, 6, 6, 6, 6])]
    x = [np.arange(16,).reshape(-1,2), np.arange(12,).reshape(-1,2)]
    # [array([[ 0,  1],
    #    [ 2,  3],
    #    [ 4,  5],
    #    [ 6,  7],
    #    [ 8,  9],
    #    [10, 11],
    #    [12, 13],
    #    [14, 15]]),
    #  array([[ 0,  1],
    #    [ 2,  3],
    #    [ 4,  5],
    #    [ 6,  7],
    #    [ 8,  9],
    #    [10, 11]])]

    data_tmp = []
    for i in range(2):
        data_tmp.append({'resp': x[i], 'labels1': labels1[i], 'labels2': labels2[i]})
    return Data(data_tmp)

def test_label_change_points_simple():
    arr = np.array([0, 0, 0, 1, 1, 3, 3])
    locs, labels, prior_labels = get_label_change_points(arr)
    assert np.array_equal(locs, np.array([3,5]))
    assert np.array_equal(labels, np.array([1,3]))
    assert np.array_equal(prior_labels, np.array([0,1]))

def test_label_change_points_negatives():
    arr = np.array([100, 100, 100, -1, -1, 2, 2])
    locs, labels, prior_labels = get_label_change_points(arr)
    assert np.array_equal(locs, np.array([3,5]))
    assert np.array_equal(labels, np.array([-1,2]))
    assert np.array_equal(prior_labels, np.array([100,-1]))

def test_label_change_points_list_error():
    arr = [100, 100, 100, -1, -1, 2, 2]
    with pytest.raises(TypeError):
        locs, labels, prior_labels = get_label_change_points(arr)

def test_label_change_points_1elem_array():
    arr = np.array([0])
    locs, labels, prior_labels = get_label_change_points(arr)
    assert np.array_equal(locs, np.array([]))
    assert np.array_equal(labels, np.array([]))
    assert np.array_equal(prior_labels, np.array([]))

def test_label_change_points_2elem_array():
    arr = np.array([-1, 2])
    locs, labels, prior_labels = get_label_change_points(arr)
    print((locs, labels, prior_labels))
    assert np.array_equal(locs, np.array(1))
    assert np.array_equal(labels, np.array(2))
    assert np.array_equal(prior_labels, np.array(-1))


# test segment_around_label_transitions

def test_single_label_segment_transitions_0prechange(outstruct):
    segments, labels, prior_labels = segment_around_label_transitions(field=outstruct['resp'], labels=outstruct['labels1'],
                                                                      prechange_samples=0,
                                                                      postchange_samples=2)
    expected = np.array([[[ 4,  5],
                          [ 6,  7]],
                         [[10, 11],
                          [12, 13]],
                         [[ 2,  3],
                          [ 4,  5]],
                         [[ 6,  7],
                          [ 8,  9]]])

    assert np.array_equal(segments, expected)
    assert np.array_equal(labels, np.array([1,3,2,0]))
    assert np.array_equal(prior_labels, np.array([0,1,-1,2]))

def test_single_label_segment_transitions_bigprechange(outstruct):
    segments, labels, prior_labels = segment_around_label_transitions(field=outstruct['resp'], labels=outstruct['labels1'],
                                                                      prechange_samples=3,
                                                                      postchange_samples=2)
    expected = np.array([[[ 4,  5],
                          [ 6,  7],
                          [ 8,  9],
                          [10, 11],
                          [12, 13]],
                         [[ 0,  1],
                          [ 2,  3],
                          [ 4,  5],
                          [ 6,  7],
                          [ 8,  9]]])
    assert np.array_equal(segments, expected)
    assert np.array_equal(labels, np.array([3,0]))
    assert np.array_equal(prior_labels, np.array([1,2]))

def test_single_label_segment_transitions_bigpostchange(outstruct):
    segments, labels, prior_labels = segment_around_label_transitions(field=outstruct['resp'], labels=outstruct['labels1'],
                                                                      prechange_samples=0,
                                                                      postchange_samples=6)
    expected = np.array([[[ 4,  5],
                          [ 6,  7],
                          [ 8,  9],
                          [10, 11],
                          [12, 13],
                          [14, 15]]])
    print((labels, prior_labels))
    assert np.array_equal(segments, expected)
    assert np.array_equal(labels, np.array([1]))
    assert np.array_equal(prior_labels, np.array([0]))

def test_single_label_segment_transitions_withlags(outstruct):
    segments, labels, prior_labels = segment_around_label_transitions(field=outstruct['resp'], labels=outstruct['labels1'],
                                                                      prechange_samples=3,
                                                                      postchange_samples=1,
                                                                      elec_lag=np.array([1,2]))
    expected = np.array([[[ 6,  9],
                          [ 8, 11],
                          [10, 13],
                          [12, 15]],
                         [[ 2,  5],
                          [ 4,  7],
                          [ 6,  9],
                          [ 8, 11]]])
    assert np.array_equal(segments, expected)
    assert np.array_equal(labels, np.array([3,0]))
    assert np.array_equal(prior_labels, np.array([1,2]))

def test_single_label_segment_transitions_withlags_multiplelabels(outstruct):
    segments, labels, prior_labels = segment_around_label_transitions(field=outstruct['resp'], labels=(outstruct['labels1'], outstruct['labels2']),
                                                                      prechange_samples=3,
                                                                      postchange_samples=1,
                                                                      elec_lag=np.array([1,2]))
    expected = np.array([[[ 6,  9],
                          [ 8, 11],
                          [10, 13],
                          [12, 15]],
                         [[ 2,  5],
                          [ 4,  7],
                          [ 6,  9],
                          [ 8, 11]]])
    labs2_ex = np.array([[3, 4, 5, 6,], [8, 7, 6, 6]])
    assert np.array_equal(segments, expected)
    assert np.array_equal(labels[0], np.array([3,0]))
    assert np.array_equal(labels[1], labs2_ex)
    assert np.array_equal(prior_labels, np.array([1,2]))
