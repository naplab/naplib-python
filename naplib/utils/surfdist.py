"""
Copied from the surfdist package https://github.com/NeuroanatomyAndConnectivity/surfdist/tree/master due to
a bug with numba versions in surfdist making it incompatible as a dependency.
"""

import gdist
import matplotlib.pyplot as plt
import numpy as np
from nibabel.freesurfer.io import read_annot


def load_freesurfer_label(annot_input, label_name):
    """
    Get source node list for a specified freesurfer label.

    Inputs
    -------
    annot_input : freesurfer annotation label file
    label_name : freesurfer label name
    cortex : not used
    """

    labels, _, names = read_annot(annot_input)
    names = [i.decode("utf-8") for i in names]
    label_value = names.index(label_name)
    label_nodes = np.array(np.where(np.in1d(labels, label_value)), dtype=np.int32)

    return label_nodes


def dist_calc(surf, cortex, source_nodes):
    """
    Calculate exact geodesic distance along cortical surface from set of source nodes.
    "dist_type" specifies whether to calculate "min", "mean", "median", or "max" distance values
    from a region-of-interest. If running only on single node, defaults to "min".
    """

    cortex_vertices, cortex_triangles = surf_keep_cortex(surf, cortex)
    translated_source_nodes = translate_src(source_nodes, cortex)
    data = gdist.compute_gdist(
        cortex_vertices, cortex_triangles, source_indices=translated_source_nodes
    )
    dist = recort(data, surf, cortex)
    del data

    return dist


def surf_keep_cortex(surf, cortex):
    """
    Remove medial wall from cortical surface to ensure that shortest paths are only calculated through the cortex.

    Inputs
    -------
    surf : Tuple containing two numpy arrays of shape (n_nodes,3). Each node of the first array specifies the x, y, z
           coordinates one node of the surface mesh. Each node of the second array specifies the indices of the three
           nodes building one triangle of the surface mesh.
           (e.g. the output from nibabel.freesurfer.io.read_geometry)
    cortex : Array with indices of vertices included in within the cortex.
             (e.g. the output from nibabel.freesurfer.io.read_label)
    """

    # split surface into vertices and triangles
    vertices, triangles = surf

    # keep only the vertices within the cortex label
    cortex_vertices = np.array(vertices[cortex], dtype=np.float64)

    # keep only the triangles within the cortex label
    cortex_triangles = triangles_keep_cortex(triangles, cortex)

    return cortex_vertices, cortex_triangles


def triangles_keep_cortex(triangles, cortex):
    """
    Remove triangles with nodes not contained in the cortex label array
    """

    # for or each face/triangle keep only those that only contain nodes within the list of cortex nodes
    input_shape = triangles.shape
    triangle_is_in_cortex = np.all(
        np.reshape(np.in1d(triangles.ravel(), cortex), input_shape), axis=1
    )

    cortex_triangles_old = np.array(triangles[triangle_is_in_cortex], dtype=np.int32)

    # reassign node index before outputting triangles
    new_index = np.digitize(cortex_triangles_old.ravel(), cortex, right=True)
    cortex_triangles = np.array(
        np.arange(len(cortex))[new_index].reshape(cortex_triangles_old.shape),
        dtype=np.int32,
    )

    return cortex_triangles


def translate_src(src, cortex):
    """
    Convert source nodes to new surface (without medial wall).
    """
    src_new = np.array(np.where(np.in1d(cortex, src))[0], dtype=np.int32)

    return src_new


def recort(input_data, surf, cortex):
    """
    Return data values to space of full cortex (including medial wall), with medial wall equal to zero.
    """
    data = np.zeros(len(surf[0]))
    data[cortex] = input_data
    return data


def surfdist_viz(
    coords,
    faces,
    stat_map=None,
    elev=0,
    azim=0,
    cmap="coolwarm",
    threshold=None,
    alpha="auto",
    bg_map=None,
    bg_on_stat=False,
    figsize=None,
    ax=None,
):
    """Visualize results on cortical surface using matplotlib.

    Parameters
    ----------
    coords : numpy array of shape (n_nodes,3), each row specifying the x,y,z
            coordinates of one node of surface mesh
    faces : numpy array of shape (n_faces, 3), each row specifying the indices
            of the three nodes building one node of the surface mesh
    stat_map : numpy array of shape (n_nodes,) containing the values to be
               visualized for each node.
    elev, azim : integers, elevation and azimuth parameters specifying the view
                 on the 3D plot. For Freesurfer surfaces elev=0, azim=0 will
                 give a lateral view for the right and a medial view for the
                 left hemisphere, elev=0, azim=180 will give a medial view for
                 the right and lateral view for the left hemisphere.
    cmap : Matplotlib colormap, the color range will me forced to be symmetric.
           Colormaps can be specified as string or colormap object.
    threshold : float, threshold to be applied to the map, will be applied in
                positive and negative direction, i.e. values < -abs(threshold)
                and > abs(threshold) will be shown.
    alpha : float, determines the opacity of the background mesh, in 'auto' mode
            alpha defaults to .5 when no background map is given, to 1 otherwise.
    bg_map : numpy array of shape (n_nodes,) to be plotted underneath the
             statistical map. Specifying a sulcal depth map as bg_map results
             in realistic shadowing of the surface.
    bg_on_stat : boolean, specifies whether the statistical map should be
                 multiplied with the background map for shadowing. Otherwise,
                 only areas that are not covered by the statsitical map after
                 thresholding will show shadows.
    figsize : tuple of intergers, dimensions of the figure that is produced.
    ax : Axis
        Axis to plot on, with 3d projection.

    Returns
    -------
    Matplotlib figure object
    """

    # load mesh and derive axes limits
    faces = np.array(faces, dtype=int)
    limits = [coords.min(), coords.max()]

    # set alpha if in auto mode
    if isinstance(alpha, str) and alpha == "auto":
        if bg_map is None:
            alpha = 0.5
        else:
            alpha = 1

    # if cmap is given as string, translate to matplotlib cmap
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    if ax is None:
        premade_ax = False
        # initiate figure and 3d axes
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", xlim=limits, ylim=limits)
    else:
        premade_ax = True
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # plot mesh without data
    p3dcollec = ax.plot_trisurf(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        triangles=faces,
        linewidth=0.0,
        antialiased=False,
        color="white",
    )

    # If depth_map and/or stat_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or stat_map is not None:
        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = 0.5 * face_colors[:, :3]

        if bg_map is not None:
            bg_data = bg_map
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError(
                    "The bg_map does not have the same number "
                    "of vertices as the mesh."
                )
            bg_faces = np.mean(bg_data[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            face_colors = plt.cm.gray_r(bg_faces)

        # modify alpha values of background
        face_colors[:, 3] = alpha * face_colors[:, 3]

        if stat_map is not None:
            stat_map_data = stat_map
            stat_map_faces = np.mean(stat_map_data[faces], axis=1)

            # Ensure symmetric colour range, based on Nilearn helper function:
            # https://github.com/nilearn/nilearn/blob/master/nilearn/plotting/img_plotting.py#L52
            vmax = max(-np.nanmin(stat_map_faces), np.nanmax(stat_map_faces))
            vmin = -vmax

            if threshold is not None:
                kept_indices = np.where(abs(stat_map_faces) >= threshold)[0]
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax - vmin)
                if bg_on_stat:
                    face_colors[kept_indices] = (
                        cmap(stat_map_faces[kept_indices]) * face_colors[kept_indices]
                    )
                else:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
            else:
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax - vmin)
                if bg_on_stat:
                    face_colors = cmap(stat_map_faces) * face_colors
                else:
                    face_colors = cmap(stat_map_faces)

        p3dcollec.set_facecolors(face_colors)

    if not premade_ax:
        return fig, ax
