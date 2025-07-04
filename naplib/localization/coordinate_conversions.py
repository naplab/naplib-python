import numpy as np
import nibabel.freesurfer.io as fsio
from scipy.spatial import cKDTree

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

def a_to_b(coords, src_pial, src_sphere, dst_pial, dst_sphere):
    """
    Convert 3D coordinates from any space to another space.
    Each subject comes with a bunch of MRI files; In this function these files are used:
    1. lh/rh.pial file of the source space      ==> SRC_PATH/surf/lh.pial
    2. lh/rh.sphere.reg file of the source      ==> SRC_PATH/surf/lh.sphere.reg
    3. lh/rh.pial file of the destination       ==> DST_PATH/surf/lh.pial
    4. lh/rh.sphere.reg file of the destination ==> DST_PATH/surf/lh.sphere.reg

    Apply this function separately for left and right hemispheres.

    NOTE: In case of converting to an atlas space, the files we need are accessible
    by installing freesurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
    Path of different atlas spaces: PATH_freesurfer/8.0.0/subjects/

    Parameters
    ----------
    coords : np.ndarray (elecs, 3)
        Coordinates in source space
    src_pial : str
        Path to the source pial surface file (e.g., 'lh.pial')
    src_sphere : str
        Path to the source sphere file (e.g., 'lh.sphere.reg')
    dst_pial : str
        Path to the destination pial surface file (e.g., 'lh.pial')
    dst_sphere : str
        Path to the destination sphere file (e.g., 'lh.sphere.reg')
    
    Returns
    -------
    new_coords : np.ndarray (elecs, 3)
        Coordinates in target space
    """
    src_sphere, _ = fsio.read_geometry(src_sphere)
    tgt_sphere, _ = fsio.read_geometry(dst_sphere)

    tree = cKDTree(tgt_sphere)

    if np.isnan(coords).any():
        print("WARNING: NaN values found in coordinates. Replacing with zeros.")
        coords = np.nan_to_num(coords)

    lh_verts_sub, _ = fsio.read_geometry(src_pial)
    lh_verts_sub_fs, _ = fsio.read_geometry(dst_pial)
    
    tree_elecs = cKDTree(lh_verts_sub)
    _, mapping_indices_elecs = tree_elecs.query(coords, k=1)

    _, mapping_indices_elecs_warped = tree.query(src_sphere[mapping_indices_elecs], k=1)

    new_coords = lh_verts_sub_fs[mapping_indices_elecs_warped]

    return new_coords
