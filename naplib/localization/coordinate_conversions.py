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

def src_to_dst(coords, src_pial, src_sphere, dst_pial, dst_sphere):
    """
    Convert 3D coordinates from any space to another space.
    Each subject comes with a bunch of MRI files; In this function these files are used:
    1. lh.pial file of the source space      ==> SRC_PATH/surf/lh.pial
    2. lh.sphere.reg file of the source      ==> SRC_PATH/surf/lh.sphere.reg
    3. lh.pial file of the destination       ==> DST_PATH/surf/lh.pial
    4. lh.sphere.reg file of the destination ==> DST_PATH/surf/lh.sphere.reg

    Provide LH files, the function assumes the RH ones are in the same directory.

    NOTE: In case of converting to an atlas space, the files we need are accessible
    by installing freesurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
    Path of different atlas spaces: PATH_freesurfer/8.0.0/subjects/

    Parameters
    ----------
    coords : np.ndarray (elecs, 3)
        Coordinates in source space. Can be in both hemispheres.
    src_pial : str/dict{'vert_lh', 'vert_rh'}
        Path to the source pial surface file (e.g., 'lh.pial'). In case of a mat file for pial surfaces,
        provide a dictionary with keys 'vert_lh' and 'vert_rh' containing the vertices for each hemisphere.
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
    src_sphere_lh, _ = fsio.read_geometry(src_sphere)
    src_sphere_rh, _ = fsio.read_geometry(src_sphere.replace('lh', 'rh'))
    src_sphere = np.vstack((src_sphere_lh, src_sphere_rh))

    tgt_sphere_lh, _ = fsio.read_geometry(dst_sphere)
    tgt_sphere_rh, _ = fsio.read_geometry(dst_sphere.replace('lh', 'rh'))

    tree_lh = cKDTree(tgt_sphere_lh)
    tree_rh = cKDTree(tgt_sphere_rh)

    if np.isnan(coords).any():
        print("WARNING: NaN values found in coordinates. Replacing with zeros.")
        coords = np.nan_to_num(coords)

    if isinstance(src_pial, str):
        lh_verts_sub, _ = fsio.read_geometry(src_pial)
        rh_verts_sub = fsio.read_geometry(src_pial.replace('lh', 'rh'))[0]
        lh_threshold = lh_verts_sub.shape[0]
        lh_verts_sub = np.vstack((lh_verts_sub, rh_verts_sub))
    else:
        lh_verts_sub = src_pial['vert_lh']
        rh_verts_sub = src_pial['vert_rh']
        lh_threshold = lh_verts_sub.shape[0]

    lh_verts_sub_fs, _ = fsio.read_geometry(dst_pial)
    rh_verts_sub_fs, _ = fsio.read_geometry(dst_pial.replace('lh', 'rh'))

    tree_elecs = cKDTree(lh_verts_sub)
    _, mapping_indices_elecs = tree_elecs.query(coords, k=1)
    
    print(f"#Electrodes in LH: {np.sum(mapping_indices_elecs < lh_threshold)}, RH: {np.sum(mapping_indices_elecs >= lh_threshold)}")

    mapping_indices_elecs_lh = mapping_indices_elecs[mapping_indices_elecs < lh_threshold]
    _, mapping_indices_elecs_warped_lh = tree_lh.query(src_sphere[mapping_indices_elecs_lh], k=1)

    mapping_indices_elecs_rh = mapping_indices_elecs[mapping_indices_elecs >= lh_threshold]
    _, mapping_indices_elecs_warped_rh = tree_rh.query(src_sphere[mapping_indices_elecs_rh - lh_threshold], k=1)

    new_coords_lh = lh_verts_sub_fs[mapping_indices_elecs_warped_lh]
    new_coords_rh = rh_verts_sub_fs[mapping_indices_elecs_warped_rh]

    new_coords = np.zeros((coords.shape[0], 3))
    new_coords[mapping_indices_elecs < lh_threshold] = new_coords_lh
    new_coords[mapping_indices_elecs >= lh_threshold] = new_coords_rh

    return new_coords