import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..localization import Brain, find_closest_vertices
from ..utils import surfdist_viz


def _view(hemi, mode: str = "lateral", backend: str = "mpl"):
    """
    Appropriate azimuth for displaying the specified hemisphere in specified view.

    Arguments
    ---------
    hemi : {'lh','rh'}
        Hemisphere
    mode : {'lateral','medial','frontal','top','best'}, default='lateral'
        What view to return azimuth and elevation for. One of 'lateral',
        'best', 'medial','frontal','top'
    backend : {'mpl','plotly'}, defualt='mpl'
        Plotting backend, either 'mpl' or 'plotly'

    Returns
    -------
    elevation : float, elevation of view, if backend='mpl'
    azimuth : float, azimuth of view, if backend='mpl'
    eye : dict, for plotly.graph_objects.layout.scene.camera.eye, if backend='plotly'
    center : dict, for plotly.graph_objects.layout.scene.camera.center, if backend='plotly'
    """
    if mode == "lateral":
        if backend == "plotly":
            eye = dict(x=-1, y=0, z=0) if hemi == "lh" else dict(x=1, y=0, z=0)
            center = dict(x=0, y=0, z=0)
            return eye, center
        else:
            return (0, 180) if hemi == "lh" else (0, 0)

    elif mode == "medial":
        if backend == "plotly":
            eye = dict(x=1, y=0, z=0) if hemi == "lh" else dict(x=-1, y=0, z=0)
            center = dict(x=0, y=0, z=0)
            return eye, center
        else:
            return (0, 0) if hemi == "lh" else (0, 180)

    elif mode == "frontal":
        if backend == "plotly":
            eye = dict(x=0, y=1, z=0)
            center = dict(x=0, y=0, z=0)
            return eye, center
        else:
            return (0, 90)

    elif mode == "top":
        if backend == "plotly":
            eye = dict(x=0, y=0, z=1)
            center = dict(x=0, y=0, z=0)
            return eye, center
        else:
            return (90, 270)

    elif mode == "best":
        if backend == "plotly":
            eye = dict(x=-1, y=0.1, z=0.1) if hemi == "lh" else dict(x=1, y=0.1, z=0.1)
            center = dict(x=0, y=0, z=0)
            return eye, center
        else:
            return (20, 160) if hemi == "lh" else (40, 20)

    raise ValueError(f"Unknown `mode`: {mode}.")


def _plot_hemi(hemi, cmap="coolwarm", ax=None, denorm=False, view="best"):
    surfdist_viz(
        *hemi.surf,
        hemi.overlay,
        *_view(hemi.hemi, mode=view),
        cmap=cmap(hemi.overlay.max()) if denorm else cmap,
        threshold=0.25,
        alpha=hemi.alpha,
        bg_map=hemi.sulc,
        bg_on_stat=True,
        ax=ax,
    )
    ax.axes.set_axis_off()
    ax.grid(False)


def plot_brain_overlay(
    brain, cmap="coolwarm", ax=None, denorm=False, view="best", **kwargs
):
    """
    Plot brain overlay on the 3D cortical surface using matplotlib.
    If certain regions have been set as visible using
    brain.set_visible(), only those regions will be shown.

    Parameters
    ----------
    brain : nl.localization.Brain
        Brain instance to plot on.
    cmap : str, default='coolwarm'
        Colormap to use.
    ax : list | tuple of matplotlib Axes
        2 Axes to plot the left and right hemispheres with.
    denorm : bool, default=False
        Whether to center the overlay labels around 0 or not before sending to the colormap.
    view : {'lateral','medial','frontal','top','best'}, default='best'
        Which view to plot for each hemisphere.
    **kwargs : kwargs
        Any other kwargs to pass to matplotlib.pyplot.figure

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple of matplotlib Axes

    """
    fig = plt.figure(**kwargs)
    if ax is None:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax = (ax1, ax2)
    else:
        ax1, ax2 = ax

    _plot_hemi(brain.lh, cmap, ax1, denorm, view=view)
    _plot_hemi(brain.rh, cmap, ax2, denorm, view=view)

    return fig, ax


def _tri_indices(simplices):
    # simplices is a numpy array defining the simplices of the triangularization
    # returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))


def _plotly_trisurf(points3D, simplices, facecolor, opacity=1, name=""):
    # points3D are coordinates of the triangle vertices
    # simplices are the simplices that define the triangularization;
    # simplices  is a numpy array of shape (no_triangles, 3)
    I, J, K = _tri_indices(simplices)

    triangles = go.Mesh3d(
        x=points3D[:, 0],
        y=points3D[:, 1],
        z=points3D[:, 2],
        facecolor=facecolor,
        i=I,
        j=J,
        k=K,
        name=name,
        opacity=opacity,
    )

    return triangles


def _plotly_scatter3d(coords, elec_colors, elec_alpha, name=""):
    marker = go.scatter3d.Marker(color=elec_colors)
    if not isinstance(elec_alpha, (np.ndarray, list)):
        elec_alpha = np.asarray([elec_alpha] * len(coords))
    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=marker,
        name=name,
        customdata=elec_alpha,
    )
    return scatter

def _set_opacity(trace):
    """https://community.plotly.com/t/varying-opacity-in-scatter-3d/75505/5"""
    if hasattr(trace, 'customdata') and isinstance(trace.customdata, float):
        opacities = trace.customdata
        r, g, b = plotly.colors.hex_to_rgb(trace.marker.color)
        trace.marker.color = [
            f'rgba({r}, {g}, {b}, {a})'
            for a in map(lambda x: x[0], opacities)]


def plot_brain_elecs(
    brain,
    elecs,
    isleft=None,
    values=None,
    colors="k",
    hemi="both",
    view="lateral",
    snap_to_surface=None,
    elec_size=4,
    cortex="classic",
    cmap="cool",
    alpha=1,
    vmin=None,
    vmax=None,
    brain_alpha=None,
    figsize=6,
    backend="mpl",
    **kwargs,
):
    """
    Plot electrodes on the brain using a simple matplotlib backend, or an interactive
    3D figure using the plotly backend.

    Due to the limitation of matplotlib being unable to render 3D surfaces
    in order as they would truly be seen by the camera angle, electrodes
    which are behind the cortical surface will still be visible as if
    they were in front of it.

    Parameters
    ----------
    brain : nl.localization.Brain
        Brain instance to plot on.
    elecs : np.ndarray
        Array of shape (num_elecs, 3) of electrode coordinates in pial space.
    isleft : np.ndarray, optional
        Boolean array of length (num_elecs) indicating whether a given electrode belongs
        to the left hemisphere. If not given, they are assumed based on the sign of the first
        component of the `elecs` coordinates (negative is left, positive is right)
    values : np.ndarray, optional
        Float values of length (num_elecs) which will be converted by the colormap colors
        for each electrode.
    colors : np.ndarray | list[str] | str, default='k'
        Colors to plot for each electrode. Ignored if values is not None. This can be a single string
        specifiying a color for all electrodes, a list of strings of the same length as
        elecs, or a numpy array of shape (num_elecs, 4) specifying the RGBA value for each electrode.
    hemi : {'both', 'lh', 'rh'}, default='both'
        Hemisphere(s) to plot. If 'both', then 2 subplots are created, one for each hemisphere.
        Otherwise only one hemisphere is displayed with its electrodes.
    view : {'lateral','frontal','medial','top','best'} | tuple, default='lateral'
        View of the brain to display. A tuple can specify the (elevation, azimuth) for matplotlib backend,
        or a tuple of dicts for (eye, center), which are the plotly.graph_objects.layout.scene.camera.eye and
        plotly.graph_objects.layout.scene.camera.center for plotly backend.
    snap_to_surface : bool, optional
        Whether to snap electrodes to the nearest point on the pial cortical surface. If plotting
        an 'inflated' brain, this should be set to True (default) to map through the pial surface,
        since coordinates are assumed to represent coordinates in the pial space. If plotting pial,
        then this can be set to False (default) to show true electrode placement, or True to map
        to the surface.
    elec_size : int | np.ndarray, default=4
        Size of the markers representing electrodes. If an array, should give the size for each electrode.
    cortex : {'classic','high_contrast','mid_contrast','low_contrast','bone'}, default='classic'
        How to map the sulci to greyscale. 'classic' will leave sulci untouched, which may be
        better for plotting the pial surface, but 'high_contrast' will enhance the contrast between
        gyri and sulci, which may be better for inflated surfaces.
    cmap : str, default='cool'
        Colormap for electrode values if values are provided.
    alpha : float | np.ndarray, optional, default=1
        Opacity of the electrodes. Either a single float or an array of same length as number of electrodes.
        If None, then the colors provided should be an array of RGBA values, not just RGB.
    vmin : float, optional
        Minimum value for colormap normalization. If None, uses the min of valus.
    vmax : float, optional
        Maximum value for colormap normalization. If None, uses the max of values.
    brain_alpha : float, optional
        Opacity of the brain surface. The default sets this to a reasonable value based on
        both the surface type ('pial' or not) and the backend ('plotly' vs 'mpl')
    figsize : int, default=6
        Size of the figure to display. This will be multiplied by the number of hemispheres
        to plot to specify the width.
    backend : {'mpl','plotly'}, default='mpl'
        Backend used for plotting. Matplotlib produces a figure and list of axes, each with a hemisphere
        as a static image. Plotly produces a 3D plot figure widget which can be saved as an image or interacted
        with in HTML form.
    **kwargs
        Additional keyword arguments to be passed to the matplotlib.pyplot.scatter function.

    Returns
    -------
    fig : matplotlib Figure or plotly Figure, depending on backend
    axes : list of matplotlib Axes, only if using mpl backend, otherwise None

    Notes
    -----
    When using within a jupyter notebook, it may be required to run these lines in this specific order before
    doing any plotting with the plotly backend.

    >>> import plotly.offline as pyo
    >>> pyo.init_notebook_mode(connected=True)
    >>> import plotly.io as pio
    >>> pio.renderers.default = 'iframe' # or possibly 'notebook'

    If plots still don't show in the notebook, you may need to install nbformat if you do not have it, or
    you may need to enable certain ipywidgets for displaying plotly in notebooks.
    Regardless, it should still be possible to save the the figure as either HTML or
    static image file. See `plotly examples documentation <https://plotly.com/python/interactive-html-export/>`_ for
    details on saving figures.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from naplib.localization import Brain
    >>> from naplib.visualization import plot_brain_elecs, plot_brain_overlay
    >>> brain = Brain('pial', subject_dir='path/to/subjects/').split_hg('midpoint').split_stg().simplify_labels()
    >>> coords = np.array([[-47.281147  ,  17.026093  , -21.833099  ],
                           [-48.273964  ,  16.155487  , -20.162935  ],
                           [-51.101261  ,  13.711058  , -16.258459  ]])
    >>> values = np.array([1, 1.5, 3]) # one value per electrode for color
    >>> # plot with matplotlib
    >>> fig, axes = plot_brain_elecs(brain, coords, values=values, hemi='lh', view='lateral')
    >>> plt.show()
    >>> # plot interactive figure with plotly
    >>> fig, axes = plot_brain_elecs(brain, coords, isleft, colors=colors, backend='plotly')
    >>> fig.write_html("interactive_brain_plot.html") # save as an interactive html plot
    >>> fig.show() # show the interactive plot in the notebook

    """
    if hemi == "both":
        surfs = {"lh": brain.lh.surf, "rh": brain.rh.surf}
        sulci = {"lh": brain.lh.sulc, "rh": brain.rh.sulc}
    elif hemi == "lh":
        surfs = {"lh": brain.lh.surf}
        sulci = {"lh": brain.lh.sulc}
    elif hemi == "rh":
        surfs = {"rh": brain.rh.surf}
        sulci = {"rh": brain.rh.sulc}
    else:
        raise ValueError(f"hemi must be either both, lh, or rh, but got {hemi}")

    if isleft is None:
        isleft = elecs[:, 0] < 0

    if backend not in ["mpl", "plotly"]:
        raise ValueError(f"backend must be either mpl or plotly but got {backend}")

    if snap_to_surface is None:
        if brain.surf_type == "pial":
            snap_to_surface = False
        else:
            snap_to_surface = True

    if brain_alpha is None:
        if brain.surf_type == "pial":
            brain_alpha = 0.3
        else:
            if backend == "plotly":
                brain_alpha = 1
            else:
                brain_alpha = 0.45

    if backend == "mpl":
        kwargs.setdefault("edgecolors", "none")
        kwargs.setdefault("depthshade", False)

    fig, axes = _plot_brain_elecs_standalone(
        brain,
        surfs,
        sulci,
        elecs=elecs,
        elec_isleft=isleft,
        elec_values=values,
        snap_to_surface=snap_to_surface,
        colors=colors,
        elec_size=elec_size,
        view=view,
        cortex=cortex,
        cmap=cmap,
        elec_alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        brain_alpha=brain_alpha,
        figsize=figsize,
        backend=backend,
        **kwargs,
    )

    return fig, axes


def _plot_brain_elecs_standalone(
    brain,
    surfs,
    sulci=None,
    elecs=None,
    elec_isleft=None,
    elec_values=None,
    snap_to_surface=True,
    colors="k",
    elec_size=4,
    cortex="classic",
    cmap="cool",
    elec_alpha=1,
    view="lateral",
    brain_alpha=0.3,
    vmin=None,
    vmax=None,
    figsize=8,
    backend="mpl",
    **kwargs,
):
    colormap_map = dict(
        classic=(dict(colormap="Greys", vmin=-1, vmax=2), lambda x: x),
        high_contrast=(dict(colormap="Greys", vmin=-0.2, vmax=1.3), lambda x: x),
        mid_contrast=(dict(colormap="Greys", vmin=-1.3, vmax=1.3), np.tanh),
        low_contrast=(dict(colormap="Greys", vmin=-4, vmax=4), lambda x: x),
        grey_binary=(
            dict(colormap="Greys", vmin=-0.9, vmax=2),
            lambda x: np.where(x < 0.5, np.tanh(x - 0.2), np.tanh(x + 0.2)),
        ),
        bone=(dict(colormap="bone", vmin=-0.2, vmax=2), lambda x: x),
    )

    assert isinstance(surfs, dict)
    
    if isinstance(elec_size, list):
        elec_size = np.asarray(elec_size)

    if cortex not in colormap_map:
        raise ValueError(
            f"Invalid cortex. Must be one of {'classic','high_contrast','low_contrast','bone_r'} but got {cortex}"
        )
        
    if isinstance(elec_alpha, list) or isinstance(elec_alpha, np.ndarray):
        update_opacity_per_elec = True
    else:
        update_opacity_per_elec = False

    sulci_cmap_kwargs, sulci_cmap_nonlinearity = colormap_map[cortex]

    hemi_keys = sorted(list(surfs.keys()))

    for k in hemi_keys:
        assert k in ["lh", "rh"]

    cmap_sulci = plt.colormaps[sulci_cmap_kwargs["colormap"]]
    vmin_sulci = sulci_cmap_kwargs["vmin"]
    vmax_sulci = sulci_cmap_kwargs["vmax"]
    norm_sulci = Normalize(vmin=vmin_sulci, vmax=vmax_sulci)
    cmap_sulci_func = lambda x: cmap_sulci(norm_sulci(sulci_cmap_nonlinearity(x)))

    # create figure
    if backend == "mpl":
        if not isinstance(figsize, tuple):
            figsize = (figsize * len(hemi_keys), int(figsize * 1.2))
        fig = plt.figure(figsize=figsize)
    else:
        trace_list = []

    if elecs is not None:
        if elec_isleft is None:
            elec_isleft = np.ones((len(elecs),)).astype("bool")

        if elec_values is not None:
            # if plotting electrodes, that overrides the colormap for overlay
            cmap_overlay = plt.colormaps[cmap]
            vmin = elec_values.min() if vmin is None else vmin
            vmax = elec_values.max() if vmax is None else vmax

            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap_func = lambda x: cmap_overlay(norm(x))
            cmap_mappable = ScalarMappable(cmap=cmap_overlay, norm=norm)

    num_subfigs = len(hemi_keys)

    # do plotting on each axis if mpl, otherwise together
    axes = []
    for i, hemi in enumerate(hemi_keys):
        verts = surfs[hemi][0]
        triangles = surfs[hemi][1]
        if sulci[hemi] is not None:
            sulc = sulci[hemi]

        if isinstance(view, str):
            elev, azim = _view(hemi, mode=view, backend=backend)
        elif isinstance(view, tuple):
            elev, azim = view
        else:
            raise ValueError("Argument `view` should be a string or tuple.")

        # color by sulci
        if sulci[hemi] is not None:
            triangle_values_sulci = np.array(
                [[sulc[nn] for nn in triangles[i]] for i in range(len(triangles))]
            ).mean(1)
            colors_sulci = cmap_sulci_func(triangle_values_sulci)
        else:
            colors_sulci = np.ones((len(triangles),4))
            colors_sulci[:,:3] = 0.5
            

        if backend == "plotly":

            colors_sulci *= 255
            colors_sulci = colors_sulci.astype("int")
            
            if len(hemi_keys) == 2:
                # add some offset between hemispheres
                # if plotting both hemispheres on brain, need to offset since
                # inflated coords are centered at zero for each hemisphere in x axis
                if hemi == "lh":
                    vert_x_offset = np.array([verts[:, 0].max() + 3, 0, 0])
                    offset_verts = verts - vert_x_offset
                else:
                    vert_x_offset = np.array([verts[:, 0].min() - 3, 0, 0])
                    offset_verts = verts - vert_x_offset

            else:
                vert_x_offset = np.array([0, 0, 0])
                offset_verts = verts

            mesh = _plotly_trisurf(
                offset_verts,
                triangles,
                facecolor=colors_sulci,
                name=f"hemi-{hemi}",
                opacity=brain_alpha,
            )
            trace_list.append(mesh)
        else:  # mpl
            ax = fig.add_subplot(
                1, num_subfigs, i + 1, projection="3d", azim=azim, elev=elev
            )
            ax.set_axis_off()
            axes.append(ax)
            p3dc = ax.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2], triangles=triangles
            )
            if sulci[hemi] is not None:
                # set the face colors of the Poly3DCollection
                colors_sulci[:, -1] = brain_alpha
                p3dc.set_fc(colors_sulci)
            else:
                p3dc.set_alpha(brain_alpha)
            

        if elecs is None:
            continue

        # snap elecs to pial surface
        if snap_to_surface:
            if hemi == "lh":
                map_surf = brain.lh.surf_pial
            else:
                map_surf = brain.rh.surf_pial
            nearest_vert_indices, _ = find_closest_vertices(map_surf[0], elecs)
            if backend == "plotly":
                coords = offset_verts[nearest_vert_indices]
            else:
                coords = surfs[hemi][0][nearest_vert_indices]
        else:
            if backend == "plotly":
                coords = elecs - vert_x_offset
            else:
                coords = elecs

        # if no elec_values specified, use given colors
        if elec_values is not None:
            elec_colors = cmap_func(elec_values)
        elif colors is None:
            elec_colors = np.zeros((len(coords), 4))
            elec_colors[:, -1] = elec_alpha
        elif isinstance(colors, str):
            if isinstance(elec_alpha, (float, int)):
                elec_colors = np.asarray(
                    [mpl.colors.to_rgba(colors, elec_alpha)] * len(elec_isleft)
                )
            else:
                elec_colors = np.asarray(
                    [mpl.colors.to_rgba(colors, alph) for alph in elec_alpha]
                )
        elif isinstance(colors, list):
            if isinstance(elec_alpha, (float, int)):
                if isinstance(elec_alpha, list) or isinstance(elec_alpha, np.ndarray):
                    elec_colors = np.asarray(
                        [mpl.colors.to_rgba(cc, alph) for cc, alph in zip(colors, elec_alpha)]
                    )
                else:
                    elec_colors = np.asarray(
                        [mpl.colors.to_rgba(cc, elec_alpha) for cc in colors]
                    )
            else:
                elec_colors = np.asarray(
                    [
                        mpl.colors.to_rgba(cc, alph)
                        for cc, alph in zip(colors, elec_alpha)
                    ]
                )
        elif isinstance(colors, np.ndarray):
            elec_colors = colors.copy()
            if elec_colors.shape[1] == 4 and elec_alpha is not None:
                elec_colors[:, 3] = elec_alpha
        else:
            raise TypeError(
                "no values given, and colors could not be interpreted as either numpy array, single color string, or list of strings"
            )

        # restrict to only this hemisphere
        if hemi == "lh":
            coords = coords[elec_isleft]
            elec_colors = elec_colors[elec_isleft]
            if isinstance(elec_size, (np.ndarray)):
                elec_size_hemi = elec_size[elec_isleft]
            else:
                elec_size_hemi = elec_size
            
        else:
            coords = coords[~elec_isleft]
            elec_colors = elec_colors[~elec_isleft]
            if isinstance(elec_size, (np.ndarray)):
                elec_size_hemi = elec_size[~elec_isleft]
            else:
                elec_size_hemi = elec_size

        if backend == "plotly":
            elec_colors *= 255
            elec_colors = elec_colors.astype("int")
            scatter = _plotly_scatter3d(coords, elec_colors, elec_alpha=elec_alpha, name=f"elecs-{hemi}")
            trace_list.append(scatter)
        else:  # mpl
            x, y, z = coords.T
            ax.scatter(x, y, z, s=elec_size_hemi, c=elec_colors, **kwargs)

    if backend == "plotly":
        axis = dict(
            showbackground=False,
            showgrid=False,  # thin lines in the background
            zeroline=False,  # thick line at x=0
            visible=False,  # numbers below
        )

        scene = dict(
            camera=dict(
                eye=elev,
                center=azim,
                projection=dict(
                    type="orthographic",  # perspective
                ),
            ),
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
            aspectratio=dict(x=0.8 * len(hemi_keys), y=1.3, z=1),
        )

        layout = go.Layout(
            title="Brain plot",
            width=figsize * 120,
            height=figsize * 120,
            scene=scene,
        )

        fig = go.Figure(data=trace_list, layout=layout)

        if update_opacity_per_elec:
            fig.for_each_trace(_set_opacity)

        # change electrode size to custom size if specified
        if elec_size is not None:
            fig = fig.for_each_trace(
                lambda trace: trace.update(marker_size=elec_size)
                if "elecs" in trace.name
                else (),
            )

        return fig, None

    if elec_values is not None:  # mpl
        fig.subplots_adjust(right=0.95)  # create space on the right hand side
        ax_cbar = plt.axes([0.96, 0.55, 0.02, 0.3])  # add a small custom axis
        fig.colorbar(mappable=cmap_mappable, cax=ax_cbar)
        axes.append((ax_cbar))

    return fig, axes
