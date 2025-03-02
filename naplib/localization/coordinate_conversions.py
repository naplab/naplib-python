import numpy as np

def mni152_to_fsaverage(coords):
    """
    Convert 3D coordinates from MNI152 space to fsaverage space.

    Parameters
    ----------
    coords : np.ndarray (elecs, 3)
        Coordinates in MNI space
    
    Returns
    -------
    new_coords : np.ndarray (elecs, 3)
        Coordinates in fsaverage space
    """
    old_coords = np.concatenate([coords, np.ones((len(coords),1))], axis=1).T
    xform = np.array([[1.0022, 0.0071, -0.0177, 0.0528], [-0.0146, 0.999, 0.0027, -1.5519], [0.0129, 0.0094, 1.0027, -1.2012]])
    new_coords = (xform @ old_coords).T
    return new_coords

def fsaverage_to_mni152(coords):
    """
    Convert 3D coordinates from fsaverage space to MNI152 space.

    Parameters
    ----------
    coords : np.ndarray (elecs, 3)
        Coordinates in fsaverage space
    
    Returns
    -------
    new_coords : np.ndarray (elecs, 3)
        Coordinates in MNI152 space
    """
    old_coords = np.concatenate([coords, np.ones((len(coords),1))], axis=1).T
    xform = np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496], [-0.013, -0.0093, 0.9971, 1.184]])
    new_coords = (xform @ old_coords).T
    return new_coords


