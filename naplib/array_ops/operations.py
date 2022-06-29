from copy import deepcopy
import numpy as np

def _extract_windows_vectorized(arr, clearing_time_index, max_time, sub_window_size):
    '''
    Vectorized method to extract sub-windows of an array.
    '''
    start = clearing_time_index + 1 - sub_window_size + 1
    
    sub_windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time + 1), 0).T
    )
    return arr[sub_windows]

def sliding_window(arr, window_len, window_key_idx=0, fill_out_of_bounds=True, fill_value=0):
    '''
    Extract windows of length window_len and put them into an array. Can be
    used for causal, anticausal, or noncausal windowing.
    
    Parameters
    ----------
    arr : shape (time, *feature_dims)
        Data to be windowed. Windowing is only applied across first dimension,
        which is assumed to be time. All other dimensions are kept the same for
        the output.
    
    window_len : int
        length of sliding window
        
    window_key_idx : int, default=0 (must be from 0 to window_len-1)
        Key point of a given sliding window. A value of 0 corresponds to causal sliding
        windows, where the first window_len-1 values in the nth window
        happen before the nth point in arr. A value of window_len corresponds to
        anti-causal sliding windows, where the first value in the nth window is
        arr[n], and the remaining window_len-1 values come after that point. A value
        of 1 would return windows where the nth window is a window starting at
        arr[n-(window_len-2)] and ending at (and including) arr[n+1].
    
    fill_out_of_bounds : bool, default=True
        If True, prepends fill_value to the first (window_len-1) samples before
        the beginning of the array across all feature dimensions
        so that the output is the same length as the input 
        (i.e. there is one window for each time point in the
        original array, though the first window will contain only zeros except
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
            arr = np.concatenate([fill_value*np.ones([window_len-1-window_key_idx, *arr.shape[1:]]), arr], axis=0)
        elif window_key_idx == window_len - 1:
            arr = np.concatenate([arr, fill_value*np.ones([window_key_idx, *arr.shape[1:]])], axis=0)
        elif window_key_idx < window_len - 1:
            arr = np.concatenate([fill_value*np.ones([window_len-1-window_key_idx, *arr.shape[1:]]), arr, fill_value*np.ones([window_key_idx, *arr.shape[1:]])], axis=0)
        else:
            raise ValueError(f'window_key_idx must be an integer from 0 to window_len-1, but got {window_key_idx}')
        
    
    return _extract_windows_vectorized(arr, window_len-2, arr.shape[0]-window_len, window_len)

def concat_apply(data_list, function, axis=0, function_kwargs=None):
    '''
    Apply a function to a list of data by first contatenating the
    list into a single array along the `axis` dimension, passing it into the function,
    and then spreading the result back into the same size list.
    The function must return an array with the `axis` dimension unchanged.
    
    Parameters
    ----------
    data_list : list of np.array's
        Each array in the list must match in all dimensions except for `axis` so
        that they can be concatenated along that dimension.
        
    function : Callable
        A function which operates on an array. It must return an array where the
        `axis` dimensions is unchanged. For example, this could be something like 
        sklearn.manifold.TSNE().fit_transform if `axis=0`, or your own custom function.
    
    axis : int, default=0
        Axis over which to concatenate and then re-split the data_list before
        and after applying the function.

    function_kwargs : dict, default=None
        If provided, a dict of keyword arguments to pass to the function.

    Returns
    -------
    output : list of np.ndarray's
        List of arrays after chopping up the output of the function into arrays
        of the same length as the original input.
    
    Examples
    --------
    '''
    lengths = np.array([x.shape[axis] for x in data_list])
    data_cat = np.concatenate(data_list, axis=axis)
    
    if function_kwargs is None:
        function_kwargs = {}
    if not isinstance(function_kwargs, dict):
        raise TypeError(f'function_kwargs must be a dict of keyword arguments, but got {type(function_kwargs)}')

    func_output = function(data_cat, **function_kwargs)

    # split output back into list, but cut off the last because it is an empty array
    output = [x for x in np.split(func_output, np.cumsum(lengths), axis=axis)[:-1]]
    
    return output
