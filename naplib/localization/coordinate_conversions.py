import numpy as np
import nibabel.freesurfer.io as fsio
from scipy.spatial import cKDTree
import nibabel as nib

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

def src_to_dst(coords, src_pial, src_sphere, dst_pial, dst_sphere, require_lh_mask=False, threshold=100, distance_report=False, verbose=False):
    """
    Convert 3D coordinates from any space to another space.
    Each subject comes with a bunch of MRI files; In this function these files are used:
    1. lh.pial file of the source space      ==> SRC_PATH/surf/lh.pial
    2. lh.sphere.reg file of the source      ==> SRC_PATH/surf/lh.sphere.reg
    3. lh.pial file of the destination       ==> DST_PATH/surf/lh.pial
    4. lh.sphere.reg file of the destination ==> DST_PATH/surf/lh.sphere.reg

    fsLR is also supported: files ending with .gii

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
    require_lh_mask : bool, optional
        If True, returns a mask indicating which coordinates are in the left hemisphere. Default is False.
    threshold : float, optional
        Maximum distance in mm for an electrode to be considered in cortex (not in depth).
    distance_report : bool, optional
        If True, returns the distance of each coordinate to the nearest vertex. Default is False.
    verbose : bool, optional
        If True, prints additional information about the conversion process. Default is False.
    
    Returns
    -------
    new_coords : np.ndarray (elecs, 3)
        Coordinates in target space
    lh_mask : np.ndarray (elecs,)
        Mask indicating which coordinates are in the left hemisphere (if `require_lh_mask` is True)
    distance : np.ndarray (elecs,) or None
        Distance of each coordinate to the nearest vertex in the source pial surface (if `distance_report` is True)
    """

    if not src_sphere.endswith('.surf.gii'):
        src_sphere_lh, _ = fsio.read_geometry(src_sphere)
        src_sphere_rh, _ = fsio.read_geometry(src_sphere.replace('lh', 'rh'))
        src_sphere = np.vstack((src_sphere_lh, src_sphere_rh))
    else:
        sphere_lh = nib.load(src_sphere)
        src_sphere_lh = sphere_lh.darrays[0].data
        sphere_rh = nib.load(src_sphere.replace('.L.', '.R.'))
        src_sphere_rh = sphere_rh.darrays[0].data
        src_sphere = np.vstack((src_sphere_lh, src_sphere_rh))

    if not dst_sphere.endswith('.surf.gii'):
        tgt_sphere_lh, _ = fsio.read_geometry(dst_sphere)
        tgt_sphere_rh, _ = fsio.read_geometry(dst_sphere.replace('lh', 'rh'))
    else:
        sphere_lh = nib.load(dst_sphere)
        tgt_sphere_lh = sphere_lh.darrays[0].data
        sphere_rh = nib.load(dst_sphere.replace('.L.', '.R.'))
        tgt_sphere_rh = sphere_rh.darrays[0].data

    tree_lh = cKDTree(tgt_sphere_lh)
    tree_rh = cKDTree(tgt_sphere_rh)

    nan_list = np.zeros(coords.shape[0], dtype=bool)
    if np.isnan(coords).any() or (np.sum(np.abs(coords),axis=1)==0).any():
        print(f"WARNING: number of NaN values found in coordinates: {np.sum(np.isnan(coords))}.")
        nan_list = np.isnan(coords)
        coords = np.nan_to_num(coords)

    if isinstance(src_pial, str):
        if not src_pial.endswith('.surf.gii'):
            lh_verts_sub, _ = fsio.read_geometry(src_pial)
            rh_verts_sub = fsio.read_geometry(src_pial.replace('lh', 'rh'))[0]
            lh_threshold = lh_verts_sub.shape[0]
            lh_verts_sub = np.vstack((lh_verts_sub, rh_verts_sub))
        else:
            lh_verts_sub = nib.load(src_pial).darrays[0].data
            rh_verts_sub = nib.load(src_pial.replace('.L.', '.R.')).darrays[0].data
            lh_threshold = lh_verts_sub.shape[0]
            lh_verts_sub = np.vstack((lh_verts_sub, rh_verts_sub))
    else:
        lh_verts_sub = src_pial['vert_lh']
        rh_verts_sub = src_pial['vert_rh']
        lh_threshold = lh_verts_sub.shape[0]

    if not dst_pial.endswith('.surf.gii'):
        lh_verts_sub_fs, _ = fsio.read_geometry(dst_pial)
        rh_verts_sub_fs, _ = fsio.read_geometry(dst_pial.replace('lh', 'rh'))
    else:
        lh_verts_sub_fs = nib.load(dst_pial).darrays[0].data
        rh_verts_sub_fs = nib.load(dst_pial.replace('.L.', '.R.')).darrays[0].data

    tree_elecs = cKDTree(lh_verts_sub)
    distance, mapping_indices_elecs = tree_elecs.query(coords, k=1)

    if np.any(distance > threshold):
        print(f"WARNING: Number of in depth electrodes (distance > {threshold} mm): {np.sum(distance > threshold)}")
        new_nans = distance > threshold
        nan_list[new_nans] = True

    if verbose:
        print(f"#Electrodes in LH: {np.sum(mapping_indices_elecs < lh_threshold)}, RH: {np.sum(mapping_indices_elecs >= lh_threshold)}")

    mapping_indices_elecs_lh = mapping_indices_elecs[mapping_indices_elecs < lh_threshold]
    _, mapping_indices_elecs_warped_lh = tree_lh.query(src_sphere[mapping_indices_elecs_lh], k=1)

    mapping_indices_elecs_rh = mapping_indices_elecs[mapping_indices_elecs >= lh_threshold]
    _, mapping_indices_elecs_warped_rh = tree_rh.query(src_sphere[mapping_indices_elecs_rh], k=1)

    new_coords_lh = lh_verts_sub_fs[mapping_indices_elecs_warped_lh]
    new_coords_rh = rh_verts_sub_fs[mapping_indices_elecs_warped_rh]

    new_coords = np.zeros((coords.shape[0], 3))
    new_coords[mapping_indices_elecs < lh_threshold] = new_coords_lh
    new_coords[mapping_indices_elecs >= lh_threshold] = new_coords_rh
    lh_mask = mapping_indices_elecs < lh_threshold

    new_coords[nan_list] = np.nan

    if require_lh_mask:
        if distance_report:
            return new_coords, lh_mask, distance
        else:
            return new_coords, lh_mask
    else:
        if distance_report:
            return new_coords, distance
        else:
            return new_coords
   