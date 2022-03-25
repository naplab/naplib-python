from copy import deepcopy
import numpy as np

def _extract_windows_vectorized(arr, clearing_time_index, max_time, sub_window_size):
    start = clearing_time_index + 1 - sub_window_size + 1
    
    sub_windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time + 1), 0).T
    )
    return arr[sub_windows]

def sliding_window(data, window_len, window_key_idx=0, fill_out_of_bounds=True, fill_value=0):
    '''
    Extract windows of data of length window_len and put them into an array. Can be
    used for causal, anticausal, or noncausal windowing.
    
    Parameters
    ----------
    data : shape (time, *feature_dims)
        Data to be windowed. Windowing is only applied across first dimension,
        which is assumed to be time. All other dimensions are kept the same for
        the output.
    
    window_len : int
        length of sliding window
        
    window_key_idx : int, default=0 (must be from 0 to window_len-1)
        Key point of a given sliding window. A value of 0 corresponds to causal sliding
        windows, where the first window_len-1 values in the nth window
        happen before the nth point in data. A value of window_len corresponds to
        anti-causal sliding windows, where the first value in the nth window is
        data[n], and the remaining window_len-1 values come after that point. A value
        of 1 would return windows where the nth window is a window starting at
        data[n-(window_len-2)] and ending at (and including) data[n+1].
    
    fill_out_of_bounds : bool, default=True
        If True, prepends fill_value to the first (window_len-1) samples before
        the beginning of the data across all feature dimensions
        so that the output is the same length as the input 
        (i.e. there is one window for each time point in the
        original data, though the first window will contain only zeros except
        for the last value). If False, does not prepend zeros, so the output
        has fewer windows than the input has time points.
    
    Returns
    -------
    windowed_resps : shape (n_samples, window_len, *feature_dims)
    
    
    Examples
    --------
    >>> arr = np.arange(1,5)
    >>> slide1 = sliding_window(arr, 3)
    >>> slide2 = sliding_window(arr, 3, 0, False)
    >>> slide3 = sliding_window(arr, 3, 2)
    >>> slide4 = sliding_window(arr, 3, 1)
    >>> print(slide1)
    [[0. 0. 1.]
     [0. 1. 2.]
     [1. 2. 3.]
     [2. 3. 4.]]
    >>> print(slide2)
    [[1 2 3]
     [2 3 4]]
    >>> print(slide3)
    [[1. 2. 3.]
     [2. 3. 4.]
     [3. 4. 0.]
     [4. 0. 0.]]
    >>> print(slide4)
     [[0. 1. 2.]
     [1. 2. 3.]
     [2. 3. 4.]
     [3. 4. 0.]]
    
    '''
    
    if fill_out_of_bounds:
        if window_key_idx == 0:
            data = np.concatenate([fill_value*np.ones([window_len-1-window_key_idx, *data.shape[1:]]), data], axis=0)
        elif window_key_idx == window_len - 1:
            data = np.concatenate([data, fill_value*np.ones([window_key_idx, *data.shape[1:]])], axis=0)
        elif window_key_idx < window_len - 1:
            data = np.concatenate([fill_value*np.ones([window_len-1-window_key_idx, *data.shape[1:]]), data, fill_value*np.ones([window_key_idx, *data.shape[1:]])], axis=0)
        else:
            raise ValueError(f'window_key_idx must be an integer from 0 to window_len-1, but got {window_key_idx}')
        
    
    return _extract_windows_vectorized(data, window_len-2, data.shape[0]-window_len, window_len)
