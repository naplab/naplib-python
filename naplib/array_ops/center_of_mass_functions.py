import numpy as np
from naplib import logger

def interp_axis(new_x, x, y, axis=0):
    """
    Perform 1D interpolation along a specified axis of a multidimensional array.

    Parameters
    ----------
    new_x : np.ndarray
        1D array of new x values for interpolation.
    x : np.ndarray
        1D array of original x values.
    y : np.ndarray
        Array of shape (..., N, ...) containing original data.
    axis : int, default=0
        Axis along which to perform interpolation (default is 0).

    Returns
    -------
    y_interp : np.ndarray
        New y array with the ``axis`` having been interpolated
    """
    # Swap axes to move the interpolation axis to the front
    y = np.swapaxes(y, axis, 0)
    
    # Define the shape of the interpolated_values array
    shape = list(y.shape)
    shape[0] = len(new_x)
    interpolated_values = np.empty(shape)
    
    # Perform interpolation along the first axis (now the interpolation axis)
    for idx in np.ndindex(y.shape[1:]):
        slices = (slice(None),) + idx
        interpolated_values[slices] = np.interp(new_x, x, y[slices])
    
    # Swap axes back to the original positions
    interpolated_values = np.swapaxes(interpolated_values, axis, 0)
    
    return interpolated_values


def center_of_mass(*args, axis=0, interp_n=None):
    '''
    Compute Center of Mass over an axis in an array.
    
    Parameters
    ----------
    x : np.ndarray, optional
        Sorted 1D array of current uneven sampling of x values for the axis over
        which to compute center of mass. For example, the current
        sampling may be at x values [1,2,5,10], so before the center
        of mass is computed, it will be interpolated along that axis.
        If this is not provided, it is assumed to be integers from 0 to the
        length of the axis.
    y : np.ndarray
        Array to compute center of mass over an axis.
    axis : int, default=0
        Axis over which to compute center of mass.
    interp_n : int, default=None
        Number of values to interpolate if currently using uneven sampling.
        If None, then new sampling will be every integer between the minimum x
        and the maximum x, so that the computed center of mass is valid
        automatically. However, if the current x values are floats and do not
        span a good range for interpolation (e.g. the x values are [0.2, 0.3, 0.6]),
        then this should be an integer like 10 instead to produce 10 samples between
        0.2 and 0.6.
    
    Returns
    -------
    x_vals : np.ndarray
        X values, where the center of mass is with respect to these x values. This is
        used for when the uneven sampling of the original x values was over floats that
        do not translate well to integer samples. So if the output ``com_indices`` is 1.333
        and the ``x_vals`` are [0.1,0.2,0.3,0.4,0.5], then the center of mass is a third of
        the way between 0.2 and 0.3, so 2.33
    com_indices : np.ndarray
        Array with same shape as y except missing the ``axis`` dimension
        which has been collapsed into a single center of mass value.
    
    '''
    if len(args) == 1:
        y = args[0]
        assert axis < y.ndim, f'bad axis, array only has {y.ndim} dimensions but got axis={axis}'
        xnew = np.arange(y.shape[axis])
    elif len(args) == 2:
        x, y_orig = args
        assert axis < y_orig.ndim, f'bad axis, array only has {y_orig.ndim} dimensions but got axis={axis}'
        assert len(x) == y_orig.shape[axis], f'The axis for interpolation has the wrong number of values for the x-values. Axis is length {y_orig.shape[axis]} but x-values are length {len(x)}'
        if interp_n is None:
            if int(x[-1]-x[0])+1 < 5:
                logger.warning(f'interp_n is None, but original sampling will create only {int(x[-1]-x[0])+1} samples when interpolating, which is low. It is recommended to set interp_n to a higher value.')
            xnew = np.linspace(x[0], x[-1], num=int(x[-1]-x[0])+1)
        else:
            xnew = np.linspace(x[0], x[-1], num=interp_n)
        y = interp_axis(xnew, np.asarray(x), y_orig, axis=axis)
    else:
        raise ValueError('Must provide either x or x, y as args')
    
    # Swap axes to move the com axis to the end
    y = _move_axis_to_end(y, axis)
    y_com = _compute_com_over_last_axis(y)

    return xnew, y_com

def _compute_com_over_last_axis(x):
    '''Compute center of mass along the last axis of an n-dimensional array.'''
    x_flattened = x.reshape(-1, x.shape[-1])
    x2 = x_flattened - x_flattened.min(axis=-1, keepdims=True) + 1 # must be above zero in case get divide by zero issue
    weights = np.arange(x2.shape[-1])
    com = (x2 * weights).sum(axis=-1) / x2.sum(axis=-1)
    return com.reshape(x.shape[:-1])

def _move_axis_to_end(arr, axis):
    """
    Move a given axis of a numpy array to the end.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Input array.
    axis : int
        The axis to be moved to the end.
    
    Returns
    -------
    arr2 : numpy.ndarray
        Array with the specified axis moved to the end.
    """
    # Calculate the new order of axes
    new_order = list(range(arr.ndim))
    new_order.pop(axis)
    new_order.append(axis)
    
    # Transpose the axes to match the desired order
    arr_reordered = np.transpose(arr, new_order)
    
    return arr_reordered
