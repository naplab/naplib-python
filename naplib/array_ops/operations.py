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
    arr : np.ndarray, shape (time, *feature_dims)
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
    windows : np.ndarray, shape (n_samples, window_len, *feature_dims...)
        Windowed array segments.
    
    
    Examples
    --------
    >>> import numpy as np
    >>> from naplib.array_ops import sliding_window
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
    
    Raises
    ------
    RuntimeError: If the callable function changes the size of the concatenation/splitting axis.
    
    Examples
    --------
    >>> import numpy as np
    >>> from naplib.array_ops import concat_apply
    >>> data = [np.arange(20).reshape((5,4)), np.arange(20, 40).reshape((5,4))] # 2 trials, 5 samples with 4 channels
    >>> data
    [array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19]]),
     array([[20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35],
            [36, 37, 38, 39]])]

    >>> # We can use PCA to reduce the channel dimensionality by fitting PCA on the
    >>> # concatenated data, transforming it, and then splitting it back into 2 trials
    >>> from sklearn.decomposition import PCA
    >>> data_pca = concat_apply(data, PCA(2).fit_transform)
    >>> data_pca
    [array([[-3.60000000e+01,  8.63623587e-15],
            [-2.80000000e+01, -2.36903429e-15],
        [-2.00000000e+01, -1.34899193e-15],
            [-1.20000000e+01, -5.15542367e-16],
            [-4.00000000e+00, -4.16724783e-16]]),
     array([[4.00000000e+00, 4.16724783e-16],
            [1.20000000e+01, 5.15542367e-16],
            [2.00000000e+01, 1.34899193e-15],
            [2.80000000e+01, 2.36903429e-15],
            [3.60000000e+01, 3.01589107e-15]])]

    >>> # We can downsample the channel dimension, making use of
    >>> # the function_kwargs parameter
    >>> from scipy.signal import resample
    >>> downsampled_channels = concat_apply(data, resample, function_kwargs={'num': 3, 'axis': 1})
    >>> downsampled_channels
    [array([[ 0.5      ,  1.1339746,  2.8660254],
            [ 4.5      ,  5.1339746,  6.8660254],
            [ 8.5      ,  9.1339746, 10.8660254],
            [12.5      , 13.1339746, 14.8660254],
            [16.5      , 17.1339746, 18.8660254]]),
     array([[20.5      , 21.1339746, 22.8660254],
            [24.5      , 25.1339746, 26.8660254],
            [28.5      , 29.1339746, 30.8660254],
            [32.5      , 33.1339746, 34.8660254],
            [36.5      , 37.1339746, 38.8660254]])]
    '''
    lengths = np.array([x.shape[axis] for x in data_list])
    data_cat = np.concatenate(data_list, axis=axis)
    
    N = data_cat.shape[axis]
    
    if function_kwargs is None:
        function_kwargs = {}
    if not isinstance(function_kwargs, dict):
        raise TypeError(f'function_kwargs must be a dict of keyword arguments, but got {type(function_kwargs)}')

    func_output = function(data_cat, **function_kwargs)
    
    if func_output.shape[axis] != N:
        raise RuntimeError(f'The callable function changed the size of the concatenation and splitting axis from {N} to {func_output.shape[axis]}, but this dimension must remain constant.')

    # split output back into list, but cut off the last because it is an empty array
    output = [x for x in np.split(func_output, np.cumsum(lengths), axis=axis)[:-1]]
    
    return output
