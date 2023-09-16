import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..stats import discriminability
from ..utils import _parse_outstruct_args

def get_label_change_points(x):
    '''
    Find the indices where x changes from one categorical (integer) value to another.
    For example, if x is [0, 0, 0, 1, 1, 3, 3], locations returned will be [3, 5]
    
    Parameters
    ----------
    x : np.ndarray of shape (time,)
        Label vector over time.
    
    Returns
    -------
    locs : array of shape (n_changes, )
        Indices where labels change.
    labels : array of shape (n_changes, )
        New label (from input `x`) after each transition.
    prior_labels : array of shape (n_changes, )
        Old label (from input `x`) before each transition.
    '''

    if not isinstance(x, np.ndarray):
        assert TypeError(f'input must be numpy array but got {type(x)}')

    diff_x = np.concatenate([np.array([0]), np.diff(x)])
    locs = np.argwhere(diff_x!=0).squeeze()
    labels = x[locs]
    labels_prior = x[locs-1]
    return locs, labels, labels_prior

def segment_around_label_transitions(data=None, field=None, labels=None, prechange_samples=50, postchange_samples=300, elec_lag=None):
    '''
    Cut x around the transition points given by the changes in the labels.
    
    Parameters
    ----------
    data : Data, optional
        Data object containing data to be normalized in one of the field. If not given, must give
        the data to be normalized directly as the ``data`` argument. 
    field : list or array-like, or a string specifying a field of the Data object
        If a string, must specify a field of the Data object passed to the ``data`` parameter.
        Field to be segmented based on the labels. field[i] is shape (time, electrode, *, ...)
    labels : string, or list of np.ndarrays or array-like with the same length as field, or a tuple of lists or array-likes each the same length as field
        If a string, specifies a field of the oustruct containing labels to use for
        segmenting. If a single list or np.ndarray (not a tuple), then labels[i]
        is shape (time, ) giving integer label of each trial.
        If a tuple, then labels[0][i] is shape (time, ) giving integer label of each sample, and the
        transition points of these labels are used to segment. The other lists of labels (like labels[1])
        are not used for segmentation but their values surrounding the transition point are returned in a tuple of labels
    prechange_samples : int, default=50
        Number of samples to include in each segment that come before a label transition.
    postchange_samples : int, default=300
        Number of samples to include in each segment that come after a label transition.
    elec_lag : list or array-like, default=None
        Provides the lag (in samples) of each electrode, must be length=n_electrodes=data[i].shape[1].
        Only used if not None. This can be computed with many methods, for example, using
        ``nl.segmentation.electrode_lags_fratio``.
        
    Returns
    -------
    segments : array of shape (n_segments, time, electrode, *, ...)
        Segments cut from data surrounding label transition points.
    labels : array-like
        If input labels is just a list, then is of shape (n_segments,) providing new label after transition point
        If input labels is a tuple of lists, then this is a tuple of array-likes, where the first is the
        same as described above, and the others are arrays of shape (n_segments, time)
    prior_labels : array-like, shape (n_segments,)
        Gives label that came before the transition in the main labels array used for segmentation

    Examples
    --------
    >>> from naplib.segmentation import segment_around_label_transitions
    >>> # In this example, we imagine we only have 1 trial of data, so all the lists of data and labels
    >>> # are length 1. In practice, they could be any length though.
    >>> arr = np.arange(20).reshape(10,2) # array of shape (time, features/electrodes)
    >>> arr
    array([[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7],
           [ 8,  9],
           [10, 11],
           [12, 13],
           [14, 15],
           [16, 17],
           [18, 19]])
    >>> # use a label of categorical values to segment the array based on
    >>> # transitions in the categorical label
    >>> label = np.array([0,0,1,1,1,0,0,3,3,3])
    >>> segments, labels, prior_labels = segment_around_label_transitions(field=[arr], labels=[label],
    ...                                                                   prechange_samples=0,
    ...                                                                   postchange_samples=3)
    >>> segments
    array([[[ 4,  5],
            [ 6,  7],
            [ 8,  9]],
           [[10, 11],
            [12, 13],
            [14, 15]],
           [[14, 15],
            [16, 17],
            [18, 19]]])
    >>> labels, prior_labels
    (array([1, 0, 3]), array([0, 1, 0]))
    >>> # We can also get the full segment value of these labels or another set of
    >>> # labels around the transition, rather than just the single values of the
    >>> # labels at the transition point, simply by passing other label vectors
    >>> # that we want to be segmented as the 2nd through nth value in a tuple of labels
    >>> label2 = np.array([0,1,2,3,4,5,6,7,8,9]) # another set of labels of interest
    >>> label_tuple = ([label], [label], [label2])
    >>> segments, labels, prior_labels = segment_around_label_transitions(field=[arr], labels=label_tuple, prechange_samples=0, postchange_samples=3)
    >>> labels # this time we get a tuple of 3 labels. The first is the same as before, and the others are fully segmented 
    (array([1, 0, 3]),
     array([[1, 1, 1],
            [0, 0, 3],
            [3, 3, 3]]),
     array([[2, 3, 4],
            [5, 6, 7],
            [7, 8, 9]]))

    '''

    x, labels = _parse_outstruct_args(data, field, labels)

    if any(dd is None for dd in x):
        raise ValueError(f'None found in field')

    if any(lab is None for lab in labels):
        raise ValueError(f'None found in labels')

    
    if isinstance(labels, tuple):
        labels, other_labels = labels[0], labels[1:]
        return_multiple_labels = True
    else:
        other_labels = (labels,)
        return_multiple_labels = False

    segments = []
    new_labels = []
    prior_labels = []
    other_labels_to_return = [[] for _ in range(len(other_labels))]
    for i, tmp_unpack in enumerate(zip(x, labels, *other_labels)):
        x_i, labels_i, other_labels_i = tmp_unpack[0], tmp_unpack[1], tmp_unpack[2:]
        label_changepoints, labels_at_changepoints, labels_before_changepoints = get_label_change_points(labels_i)
        
        label_changepoints = label_changepoints.astype('int')
        labels_before_changepoints = labels_before_changepoints.astype('int')
        
        for change_point, new_lab, prior_lab in zip(label_changepoints, labels_at_changepoints, labels_before_changepoints):
            if change_point >= prechange_samples and change_point+postchange_samples <= x_i.shape[0]:
                if elec_lag is not None:
                    tmp_x_i_region = np.array([x_i[change_point-prechange_samples+elec_lag[elec_idx]:change_point+postchange_samples+elec_lag[elec_idx], elec_idx] for elec_idx in range(len(elec_lag))])
                    segments.append(tmp_x_i_region.transpose())
                else:
                    segments.append(x_i[change_point-prechange_samples:change_point+postchange_samples])
                new_labels.append(new_lab)
                prior_labels.append(prior_lab)

                for i, other_labs in enumerate(other_labels_i):
                    other_labels_to_return[i].append(other_labs[change_point-prechange_samples:change_point+postchange_samples])
                    
    if return_multiple_labels:
         return np.array(segments), (np.array(new_labels), *[np.array(tt) for tt in other_labels_to_return]), np.array(prior_labels)
    else:
        return np.array(segments), np.array(new_labels), np.array(prior_labels)
    
def electrode_lags_fratio(data=None, field=None, labels=None, max_lag=20, return_fratios=False):
    '''
    Compute lags of each electrode based on peak of f-ratio to a given label, such as
    phoneme labels. The data is segmented around onset transitions in the labels, and
    an electrode's lag is defined as the peak of the f-ratio after the transition.
    This uses the "lda" method of ``naplib.stats.discriminability``.
    
    Parameters
    ----------
    data : Data, optional
        Data containing data to be normalized in one of the field. If not given, must give
        the data to be normalized directly as the ``data`` parameter. 
    field : string, or list or array-like
        Electrode data to use to compute lags with the fratio.
        If a string, must specify a field of the Data object passed to the ``data`` parameter.
        If a list or array-like, field[i] is shape (time, n_electrodes, *, ...)
    labels : string, or list or array-like
        Labels over time for each trial. The onset changes of these labels will
        be used to segment the data and compute the f-ratio after the transition.
        If a string, must specify a field of the Data object. If a list or array-like, then
        labels[i] is of shape (time,)
    max_lag : int, default=20
        Maximum lag to look for in the f-ratio (in samples).
    return_fratios : bool, default=False
        Whether or not to return smoothed_fratios along with lags in a tuple. Default False.
    
    Returns
    -------
    lags : np.ndarray, shape=(n_electrodes,)
        Lag of each electrode, in samples
    smoothed_fratios : np.ndarray, shape=(n_electrodes, time)
        Fratio of each electrode over time, after gaussian smoothing.
        Only returned if ``return_fratios=True``
    '''
        
    segments, labels, _ = segment_around_label_transitions(data=data, field=field, labels=labels, prechange_samples=0, postchange_samples=max_lag)
    
    fratios_lags = discriminability(segments.transpose((2,1,0)), labels, elec_mode='individual')
    
    fratios_lags_smooth = gaussian_filter1d(fratios_lags, 0.5, mode='constant')

    lags = fratios_lags_smooth.argmax(-1)
    
    if return_fratios:
        return lags, fratios_lags_smooth
    else:
        return lags

def shift_label_onsets(data=None, labels='wrd_labels', p=0.5):
    """
    From continuous labels, shift them so that they correspond to a
    certain fraction of the feature's progress. This is useful if you have something
    like word labels, which encode word onsets and offsets, but you want to be able
    to segment your data based on word centers (i.e. 50% of the way through the word).

    For example, if the labels for one trial are a continuous (discrete)
    signal representing word labels over time,
    such as [-1,-1,-1,3,3,3,3,4,4,4,-1,-1,5,5,5,5,5] where -1 means no word, and 3, 4,
    and 5 indicate different words which are present over those samples, then with
    p=0.5, the onsets will be shifted so that their onsets correspond to 50% of
    the word's progress. So, the output will be [-1,-1,-1,-1,-1,3,3,-1,4,4,-1,-1,-1,-1,5,5,5].
    
    Parameters
    ----------
    data : Data, optional
        Data instance. If provided, then labels should indicate a field of this data.
    labels : string, or list of np.ndarrays, default='wrd_labels'
        Either a field of the data, or a list of 1D numpy arrays.
    p : float (0<=p<1), default=0.5
        Initial proportion of the label duration for each category to remove. For word labels, a value
        of 0.5 will replace the first 50% of each word with -1, while a value of 0.99999999 will almost certainly
        guarantee that only the last sample of each label region remains and the rest are
        replaced with -1. 0.5 can be used to get word centers, while 0.99999999 can be used to get word
        offsets.
    
    Returns
    -------
    new_labels : list[np.ndarray]
        New labels for each trial.
    
    Examples
    --------
    >>> from naplib.segmentation import shift_label_onsets
    >>> import numpy as np
    >>> labs = [np.array([-1,-1,-1,3,3,3,3,4,4,4,-1,-1,5,5,5,5,5])]
    >>> shift_label_onsets(labels=labs)
    [array([-1, -1, -1, -1, -1,  3,  3, -1,  4,  4, -1, -1, -1, -1,  5,  5,  5])]
    
    """
    if p >= 1 or p < 0:
        raise ValueError(f'p must be in the range [0, 1), but got {p}')
        
    labels = _parse_outstruct_args(data, labels, allow_strings_without_outstruct=False)
    
    new_labels = []
    
    for labels_this_trial in labels:
        
        new_labels_this_trial = labels_this_trial.copy().squeeze()
        new_labels_this_trial = np.concatenate([np.array([-1]), new_labels_this_trial])
        
        locs, labs, prior_labs = get_label_change_points(new_labels_this_trial)

        for i, (loc, lab, prior_lab) in enumerate(zip(locs, labs, prior_labs)):
            if lab == -1:
                continue
            if i == len(locs)-1:
                feature_length = len(labels_this_trial) - loc + 1
            else:
                feature_length = locs[i+1] - loc

            remove_samples = int(p*feature_length)
            new_labels_this_trial[loc:loc+remove_samples] = -1

        new_labels.append(new_labels_this_trial[1:]) # remove extra -1 that we had prepended
    return new_labels
    
