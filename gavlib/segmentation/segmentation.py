import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from ..encoding import fratio


def get_label_change_points(x):
    '''return the indices where x changes from one categorical (integer) value to another.
    For example, if x is [0, 0, 0, 1, 1, 3, 3], return will be [0, 3, 5]
    
    x : array of shape (time, )
    '''
    
    diff_x = np.concatenate([np.array([0]), np.diff(x)])
    locs = np.argwhere(diff_x!=0).squeeze()
    labels = x[locs]
    labels_prior = x[locs-1]
    return locs, labels, labels_prior

def segment_around_labeltransitions(x, labels, prechange_samples=50, postchange_samples=300, mode='FromAll', elec_lag=None):
    '''
    Cut x around the transition points given by the changes in the labels.
    
    x : list or array-like, length 18
        x[i] is shape (time, electrode, *, ...)
    labels : list or array-like, same length as x, or tuple of lists or array-likes each the same length as x
        if not a tuple, labels[i] is shape (time, ) giving integer label of each sample
        if a tuple, then labels[0][i] is shape (time, ) giving integer label of each sample, and the
            transition points of these labels are used to segment. The other lists of labels (like labels[1])
            are not used for segmentation but their values surrounding the transition point are returned in a tuple of labels
    mode : string, one of ['FromAll','FromClean']
        if FromAll, gives all transitions, but if FromClean, then all transitions are either to clean
        or from clean to one of the noises, never from noise to noise
    elec_lag : list or array-like, default=None
        Provides the lag (in samples) of each electrode, must be length (n_electrodes)=x[i].shape[1].
        Only used if not None.
        
    Returns
    -------
    segments : array of shape (n_segments, time, electrode, *, ...)
    
    labels : array-like
        If input labels is just a list, then is of shape (n_segments,) providing new label after transition point
        If input labels is a tuple of lists, then this is a tuple of array-likes, where the first is the
            same as described above, and the others are arrays of shape (n_segments, time)
            
    prior_labels : array-like, shape (n_segments,)
        Gives label that came before the transition in the main labels array used for segmentation

    '''
    
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
        if isinstance(x_i, torch.Tensor):
            x_i = x_i.numpy()
        label_changepoints, labels_at_changepoints, labels_before_changepoints = get_label_change_points(labels_i)
        
        label_changepoints = label_changepoints.astype('int')
        good_locs = labels_at_changepoints>0
        label_changepoints = label_changepoints[good_locs]
        labels_at_changepoints = labels_at_changepoints[good_locs]
        labels_before_changepoints = labels_before_changepoints[good_locs]
        
        for i_c, (change_point, new_lab, prior_lab) in enumerate(zip(label_changepoints[:-1], labels_at_changepoints[:-1], labels_before_changepoints[:-1])):
            if change_point > prechange_samples and change_point+postchange_samples < x_i.shape[0]:
                if mode == 'FromClean':
                    if (new_lab==4) or (new_lab!=4 and prior_lab==4):
                        if elec_lag is not None:
                            tmp_x_i_region = np.array([x_i[change_point-prechange_samples+elec_lag[elec_idx]:change_point+postchange_samples+elec_lag[elec_idx], elec_idx] for elec_idx in range(len(elec_lag))])
                            segments.append(tmp_x_i_region.transpose())
                        else:
                            segments.append(x_i[change_point-prechange_samples:change_point+postchange_samples])
                        new_labels.append(new_lab)
                        prior_labels.append(prior_lab)
                        for i, other_labs in enumerate(other_labels_i):
                            other_labels_to_return[i].append(other_labs[change_point-prechange_samples:change_point+postchange_samples])
                elif mode == 'FromAll':
                    if elec_lag is not None:
                        tmp_x_i_region = np.array([x_i[change_point-prechange_samples+elec_lag[elec_idx]:change_point+postchange_samples+elec_lag[elec_idx], elec_idx] for elec_idx in range(len(elec_lag))])
                        segments.append(tmp_x_i_region.transpose())
                    else:
                        segments.append(x_i[change_point-prechange_samples:change_point+postchange_samples])
                    new_labels.append(new_lab)
                    prior_labels.append(prior_lab)
                    for i, other_labs in enumerate(other_labels_i):
                        other_labels_to_return[i].append(other_labs[change_point-prechange_samples:change_point+postchange_samples])
                    
#                     print(x_i[change_point-prechange_samples:change_point+postchange_samples].shape)
                else:
                    raise Exception(f'mode={mode} is invalid')
                    
#         print(len(segments))
    if return_multiple_labels:
         return np.array(segments), (np.array(new_labels), *[np.array(tt) for tt in other_labels_to_return]), np.array(prior_labels)
    else:
        return np.array(segments), np.array(new_labels), np.array(prior_labels)
    
def compute_electrode_lags(x, labels, max_lag=20, return_fratios=False):
    '''
    x : list or array-like, length 18
        x[i] is shape (time, n_electrodes, *, ...)
    labels : list or array-like, same length as x
    return_fratios : bool
        whether or not to return smoothed_fratios along with lags
    
    Returns
    -------
    lags : np.ndarray, shape=(n_electrodes,)
        lag of each electrode, in samples
    smoothed_fratios : np.ndarray, shape=(n_electrodes, time)
        fratio of each electrode over time, after gaussian smoothing.
        Only returned if ``return_fratios``==True
    '''
    
    segments, labels, _ = segment_around_labeltransitions(x, labels, prechange_samples=0, postchange_samples=max_lag)
    
    fratios_lags = fratio(segments.transpose((2,1,0)), labels, elec_mode='individual')
    
    fratios_lags_smooth = gaussian_filter1d(fratios_lags, 0.5, mode='constant')

    lags = fratios_lags_smooth.argmax(-1)
    
    if return_fratios:
        return lags, fratios_lags_smooth
    else:
        return lags
    