"""
Functions for localizing and plotting intracranial electrodes on the freesurfer average brain.
"""

import os
import warnings
from os.path import join as pjoin

import numpy as np
from nibabel.freesurfer.io import read_geometry, read_label, read_morph_data, read_annot
from scipy.spatial.distance import cdist
from skspatial.objects import Line, Plane
from hdf5storage import loadmat

from naplib.utils import dist_calc, load_freesurfer_label
from naplib import logger

warnings.filterwarnings("ignore", message="nopython", append=True)

HEMIS = ("lh", "rh")
SURF_TYPES = ("pial", "inflated")

num2region_D_custom = {
    # My custom labels
    76: "O_pmHG",
    77: "O_alHG",
    78: "O_Te10",
    79: "O_Te11",
    80: "O_Te12",
    81: "O_mSTG",
    82: "O_pSTG",
    83: "O_IFG",
}

num2region_DK_custom = {
    # My custom labels
    42: "O_IFG",
}

class Hemisphere:
    def __init__(
        self,
        hemi: str,
        surf_type: str = "pial",
        subject: str = "fsaverage",
        coordinate_space: str = 'FSAverage',
        atlas=None,
        subject_dir=None,
    ):
        """
        Hemisphere object

        Parameters
        ----------
        hemi : str
            Either 'lh' or 'rh'.
        surf_type : str, default='pial'
            Cortical surface type, either 'pial' or 'inflated' or another if the corresponding
            files can be found.
        subject : str, default='fsaverage'
            Subject to use, must be a directory within ``subject_dir``
        coordinate_space : str, default='FSAverage'
            Coordinate space of brain vertices. Must be 'FSAverage' or 'MNI152'
        atlas : str, default=None
            Atlas for brain parcellation. Defaults to 'Destrieux' for coordinate_space='FSAverage'
            and 'Desikan-Killiany' for 'MNI152'. Can also be an annotation file name given by
            ``{subject_dir}/{subject}/label/?h.{atlas}.annot``
        subject_dir : str/path-like, defaults to SUBJECT_DIR environment variable, or the current directory
            if that does not exist.
            Path containing the subject's folder.
        """
        if hemi not in HEMIS:
            raise ValueError(f"Argument `hemi` should be in {HEMIS}.")
        if surf_type not in SURF_TYPES:
            raise ValueError(f"Argument `surf_type` should be in {SURF_TYPES}.")
        if not atlas:
            if coordinate_space == 'FSAverage':
                atlas = 'Destrieux'
            # Use DK for MNI152 or any other
            else:
                atlas = 'Desikan-Killiany'

        self.hemi = hemi
        self.surf_type = surf_type
        self.subject = subject
        self.coordinate_space = coordinate_space

        if subject_dir is None:
            subject_dir = os.environ.get("SUBJECTS_DIR", "./")

        self.subject_dir = subject_dir

        if atlas not in ['Desikan-Killiany', 'Destrieux'] and not os.path.exists(self.label_file(f'{self.hemi}.{atlas}.annot')):
            raise ValueError('Bad atlas. Try "Desikan-Killiany" or "Destrieux"')
        self.atlas = atlas

        # Check if fsaverage geometry exists
        if self.coordinate_space == 'FSAverage':
            if os.path.exists(self.surf_file(f"{hemi}.{surf_type}")):
                self.surf = read_geometry(self.surf_file(f"{hemi}.{surf_type}"))
                self.surf_pial = read_geometry(self.surf_file(f"{hemi}.pial"))
            else:
                self.coordinate_space = 'MNI152'
                print('Trying MNI152 coordinate space')
        # Use MN152 coordinate space if not
        if self.coordinate_space == 'MNI152':
            # try to find .mat file
            surf_ = loadmat(self.surf_file(f"{hemi}_pial.mat"))
            coords, faces = surf_['coords'], surf_['faces']
            faces -= 1 # make faces zero-indexed
            self.surf = (coords, faces)
            self.surf_pial = (coords, faces)
        if self.coordinate_space not in ['FSAverage','MNI152']:
            raise ValueError(f"Argument `coordinate_space`={self.coordinate_space} not implemented.")
        
        try:
            self.cort = np.sort(read_label(self.label_file(f"{hemi}.cortex.label")))
        except Exception as e:
            logger.warning(f'No {hemi}.cortex.label file found. Assuming the entire surface is cortex.')
            self.cort = np.arange(self.surf[0].shape[0])
    
        try:
            self.sulc = read_morph_data(self.surf_file(f"{hemi}.sulc"))
        except Exception as e:
            logger.warning(f'No {hemi}.sulc file found. No sulcus information will be used.')
            self.sulc = None


        self.load_labels()
        self.reset_overlay()

    @property
    def coords(self):
        return self.surf[0]

    @property
    def n_verts(self):
        return self.surf[0].shape[0]

    @property
    def trigs(self):
        return self.surf[1]

    @property
    def n_trigs(self):
        return self.surf[1].shape[0]

    @property
    def label_names(self):
        return [self.num2label[n] for n in list(set(self.labels))]

    def surf_file(self, file: str):
        return pjoin(self.subject_dir, self.subject, "surf", file)

    def label_file(self, file: str):
        return pjoin(self.subject_dir, self.subject, "label", file)

    def other_file(self, file: str):
        return pjoin(self.subject_dir, self.subject, "other", file)

    def load_labels(self):
        """
        Load Destrieux labels of each vertex from annotation files.

        Returns
        -------
        self : instance of self
        """
        self.ignore = np.zeros(self.n_verts, dtype=bool)
        annot_file = self.label_file(f"{self.hemi}.aparc.a2005s.annot")
        if os.path.exists(annot_file):
            for reg in ("Unknown", "Medial_wall"):
                self.ignore[load_freesurfer_label(annot_file, reg)] = True

        self.labels = np.zeros(self.n_verts, dtype=int)

        if self.coordinate_space == 'MNI152':
            if self.atlas == 'Desikan-Killiany':
                annot_file = self.label_file(f"FSL_MNI152.{self.hemi}.aparc.split_STG_MTG.annot")
            else:
                annot_file = self.label_file(f'{self.hemi}.{self.atlas}.annot')

        elif self.coordinate_space  == "FSAverage":
            if self.atlas == 'Desikan-Killiany':
                annot_file = self.label_file(f"{self.hemi}.aparc.annot")
            elif self.atlas == 'Destrieux':
                annot_file = self.label_file(f"{self.hemi}.aparc.a2009s.annot")
            else:
                annot_file = self.label_file(f'{self.hemi}.{self.atlas}.annot')

        else:
            raise ValueError('Bad coordinate space')
            
        if os.path.exists(annot_file):
            _,_,regions = read_annot(annot_file)
            regions = [i.decode("utf-8") for i in regions]
            num2region = {k:v for k,v in enumerate(regions)}

            for ind, reg in num2region.items():
                self.labels[load_freesurfer_label(annot_file, reg)] = ind
        else:
            raise ValueError('Bad atlas. Try "Desikan-Killiany" or "Destrieux".')

        if self.atlas == 'Destrieux':
            num2region.update(num2region_D_custom)
        elif self.atlas == 'Desikan-Killiany' and self.coordinate_space == 'MNI152':
            num2region.update(num2region_DK_custom)

        self.labels[self.ignore] = 0
        self.num2label = num2region
        self.label2num = {v: k for k, v in self.num2label.items()}

        self.simplified = False

        self.is_mangled_hg = False
        self.is_mangled_tts = False
        self.is_mangled_stg = False
        self.is_mangled_ifg = False

        return self

    def simplify_labels(self):
        """
        Simplify Destrieux and Desikan-Killiany labels into shortforms.

        Returns
        -------
        self : instance of self
        """
        if self.atlas == 'Destrieux':
            conversions = {
                "Other": [],  # Autofill all uncovered vertecies
                "HG": ["G_temp_sup-G_T_transv"],
                "pmHG": ["O_pmHG"],
                "alHG": ["O_alHG"],
                "Te1.0": ["O_Te10"],
                "Te1.1": ["O_Te11"],
                "Te1.2": ["O_Te12"],
                "TTS": ["S_temporal_transverse"],
                "PT": ["G_temp_sup-Plan_tempo"],
                "PP": ["G_temp_sup-Plan_polar"],
                "MTG": ["G_temporal_middle"],
                "ITG": ["G_temporal_inf"],
                "STG": ["G_temp_sup-Lateral"],
                "mSTG": ["O_mSTG"],
                "pSTG": ["O_pSTG"],
                "STS": ["S_temporal_sup"],
                "IFG": ["O_IFG"],
                "IFG.opr": ["G_front_inf-Opercular"],
                "IFG.tri": ["G_front_inf-Triangul"],
                "IFG.orb": ["G_front_inf-Orbital"],
                "Subcnt": ["G_and_S_subcentral"],
                "Insula": ["G_Ins_lg_and_S_cent_ins", "G_insular_short"],
                "T.Pole": ["Pole_temporal"],
            }
            

        elif self.atlas == 'Desikan-Killiany':
            d1 = {k: [k] for k in self.label2num.keys() if k not in ['O_IFG','parsopercularis','parstriangularis','parsorbitalis']}
            d2_override = {
                "Other": [],
                "IFG": ["O_IFG"],
                "IFG.opr": ["parsopercularis"],
                "IFG.tri": ["parstriangularis"],
                "IFG.orb": ["parsorbitalis"],
            }
            conversions = {**d1, **d2_override}
        else:
            raise ValueError('Bad atlas')
            
        conversions = {
            key: [self.label2num[g] for g in groups]
            for key, groups in conversions.items()
        }
        simple_num2label = {i: k for i, k in enumerate(conversions)}
        simple_label2num = {v: k for k, v in simple_num2label.items()}

        simple_labels = np.zeros(self.n_verts, dtype=int)
        for key, groups in conversions.items():
            for g in groups:
                simple_labels[self.labels == g] = simple_label2num[key]

        self.labels = simple_labels
        self.num2label = simple_num2label
        self.label2num = simple_label2num
        self.simplified = True

        return self

    def filter_labels(self, labels):
        """
        Returns mask of vertices that are within the union of `labels`.

        Parameters
        ----------
        labels : str | list[str]
            Label(s) of zone(s) to include in the binary mask.

        Returns
        -------
        mask : boolean array of shape (n_verts,).
        """
        if isinstance(labels, str):
            labels = (labels,)

        mask = np.zeros(self.n_verts, dtype=bool)
        for label in labels:
            if label in self.label2num:
                mask[self.labels == self.label2num[label]] = True

        return mask

    def zones(self, labels, min_alpha=0):
        """
        Build zone map of brain where vertices in region-of-interest (union of `labels`), have
        alpha=1, and everywhere else has alpha=`min_alpha`.

        Parameters
        ----------
        labels : str | list[str]
            Label or labels to include in the zone.
        min_alpha : float
            Value to assign to regions not included.

        Returns
        -------
        verts : array of shape (n_verts,) with values of 0 and 1.
        trigs : array of shape (n_triangles,) with values between 0 and 1.
        zones : array of shape (n_verts,) with integer values corresponding to index of label.
        """
        if isinstance(labels, str):
            labels = (labels,)
        labels = [l for l in labels if l in self.label2num]

        verts = np.zeros(self.n_verts, dtype=bool)
        zones = np.zeros(self.n_verts, dtype=int)
        for i, label in enumerate(labels):
            nodes = self.labels == self.label2num[label]
            verts[nodes] = 1
            zones[nodes] = i + 1

        trigs = np.zeros(self.n_trigs, dtype=float)
        for i in range(self.n_trigs):
            trigs[i] = np.mean([verts[self.trigs[i, j]] != 0 for j in range(3)])
            if trigs[i] < min_alpha:
                trigs[i] = min_alpha
            # if self.ignore[self.trigs[i]].any():
            #     trigs[i] = 0

        return verts, trigs, zones

    def fit_ml_line(self, points):
        """
        Fit a mediolateral line to `points` that goes from medial to lateral end.

        Parameters
        ----------
        points : np.ndarray
            Array of 3d point coordinates in the shape of (n_points, 3).

        Returns
        -------
        line : Line, the fit line as a Line object.
        """
        line = Line.best_fit(points)
        line.direction = (
            -line.direction if line.point[0] * line.direction[0] < 0 else line.direction
        )
        return line

    def fit_ml_plane_from_line(self, points):
        """
        Fit an anterior facing plane to mediolateral `points`.

        Parameters
        ----------
        points : np.ndarray
            Array of 3d point coordinates in the shape of (n_points, 3).

        Returns
        -------
        plane : Plane, the fit plane as a Plane object.
        """
        line = Line.best_fit(points)
        plane = Plane.from_vectors(points.mean(0), line.direction, [0, 0, 1])
        plane.normal = -plane.normal if plane.normal[1] < 0 else plane.normal
        return plane

    def split_hg(self, method="midpoint"):
        """
        Split HG vertices into subregions, such as posteromedial (pmHG) and anterolateral (alHG) halves.

        Parameters
        ----------
        method : {'midpoint', 'endpoint', 'median', 'te1x', 'six_four', or 'seven_three'}, default='midepoint'
            How to split the region.

        Returns
        -------
        self : instance of self
        """
        if self.is_mangled_hg:
            raise RuntimeError(
                "HG cannot be split as it is already mangled. Try changing order of operations?"
            )
        self.is_mangled_hg = True

        if self.atlas != 'Destrieux':
            raise ValueError(f'split_hg() only supported for Destrieux atlas.')
            
        hg = self.filter_labels(["G_temp_sup-G_T_transv", "HG"])

        if method == "midpoint":
            # Fit line to HG and project vertices to line
            position = self.fit_ml_line(self.coords[hg]).transform_points(
                self.coords[hg]
            )
            midpoint = np.mean((min(position), max(position)))
            # Split HG using midpoint of line
            medial = position <= midpoint
            self.labels[np.where(hg)[0][medial]] = self.label2num[
                "pmHG" if self.simplified else "O_pmHG"
            ]
            self.labels[np.where(hg)[0][~medial]] = self.label2num[
                "alHG" if self.simplified else "O_alHG"
            ]
        elif method == "six_four" or method == "seven_three":
            # Fit line to HG and project vertices to line
            position = self.fit_ml_line(self.coords[hg]).transform_points(
                self.coords[hg]
            )
            if method == "six_four":
                midpoint = 0.4 * min(position) + 0.6 * max(position)
            else:
                midpoint = 0.3 * min(position) + 0.7 * max(position)
            # Split HG using midpoint of line
            medial = position <= midpoint
            self.labels[np.where(hg)[0][medial]] = self.label2num[
                "pmHG" if self.simplified else "O_pmHG"
            ]
            self.labels[np.where(hg)[0][~medial]] = self.label2num[
                "alHG" if self.simplified else "O_alHG"
            ]
        elif method == "endpoint":
            # Fit line to HG and find two furthest endpoints w.r.t. the line
            position = self.fit_ml_line(self.coords[hg]).transform_points(
                self.coords[hg]
            )
            medpoint, latpoint = np.argmin(position), np.argmax(position)
            # Distance of HG from endpoint 1
            dist_mp = dist_calc(self.surf, self.cort, np.where(hg)[0][medpoint])
            dist_mp[self.ignore] = np.inf
            # Distance of HG from endpoint 2
            dist_lp = dist_calc(self.surf, self.cort, np.where(hg)[0][latpoint])
            dist_lp[self.ignore] = np.inf
            # Join each point to closer endpoint
            closer_to_medpoint = (
                np.argmin(np.stack((dist_mp[hg], dist_lp[hg])), axis=0) == 0
            )
            self.labels[np.where(hg)[0][closer_to_medpoint]] = self.label2num[
                "pmHG" if self.simplified else "O_pmHG"
            ]
            self.labels[np.where(hg)[0][~closer_to_medpoint]] = self.label2num[
                "alHG" if self.simplified else "O_alHG"
            ]
        elif method == "median":
            # Fit line to HG and project vertices to line
            position = self.fit_ml_line(self.coords[hg]).transform_points(
                self.coords[hg]
            )
            midpoint = np.median(position)
            # Split HG using midpoint of line
            medial = position <= midpoint
            self.labels[np.where(hg)[0][medial]] = self.label2num[
                "pmHG" if self.simplified else "O_pmHG"
            ]
            self.labels[np.where(hg)[0][~medial]] = self.label2num[
                "alHG" if self.simplified else "O_alHG"
            ]
        elif method == "te1x":
            # Read Te1.x labels
            te10 = read_label(self.other_file(f"{self.hemi}.te10.label"))
            te11 = read_label(self.other_file(f"{self.hemi}.te11.label"))
            te12 = read_label(self.other_file(f"{self.hemi}.te12.label"))
            # Distance of HG from Te1.0
            dist_te10 = dist_calc(self.surf, self.cort, te10)
            dist_te10[self.ignore] = np.inf
            # Distance of HG from Te1.1
            dist_te11 = dist_calc(self.surf, self.cort, te11)
            dist_te11[self.ignore] = np.inf
            # Distance of HG from Te1.2
            dist_te12 = dist_calc(self.surf, self.cort, te12)
            dist_te12[self.ignore] = np.inf
            # Join each point to closest endpoint
            closest = np.argmin(
                np.stack((dist_te10[hg], dist_te11[hg], dist_te12[hg])), axis=0
            )
            self.labels[np.where(hg)[0][closest == 0]] = self.label2num[
                "Te1.0" if self.simplified else "O_Te10"
            ]
            self.labels[np.where(hg)[0][closest == 1]] = self.label2num[
                "Te1.1" if self.simplified else "O_Te11"
            ]
            self.labels[np.where(hg)[0][closest == 2]] = self.label2num[
                "Te1.2" if self.simplified else "O_Te12"
            ]
        else:
            raise ValueError(f"Invalid method argument {method}")

        return self

    def remove_tts(self, method="split"):
        """
        Convert TTS labels into either HG or PT ones.

        Parameters
        ----------
        method : {'split', 'join_hg', 'join_pt'}, default='split'
            Method for removing. 'split' will convert labels to either PT or HG depending on
            which is closer, while 'join_hg' or 'join_pt' will convert the entire region to
            HG or PT, respectively.

        Returns
        -------
        self : instance of self
        """
        if self.atlas != 'Destrieux':
            raise ValueError(f'remove_tts() only supported for Destrieux atlas.')
        
        if self.is_mangled_tts:
            raise RuntimeError(
                "TTS cannot be removed as it is already mangled. Try changing order of operations?"
            )
        self.is_mangled_tts = True

        tts = self.filter_labels(["S_temporal_transverse", "TTS"])

        if method == "join_hg":
            self.labels[tts] = self.label2num[
                "HG" if self.simplified else "G_temp_sup-G_T_transv"
            ]
        elif method == "join_pt":
            self.labels[tts] = self.label2num[
                "PT" if self.simplified else "G_temp_sup-Plan_tempo"
            ]
        elif method == "split":
            # Distance of TTS points from HG
            hg = self.filter_labels(
                ["G_temp_sup-G_T_transv", "HG", "O_pmHG", "O_alHG", "pmHG", "alHG"]
            )
            dist_hg = dist_calc(self.surf, self.cort, np.where(hg)[0])
            dist_hg[self.ignore] = np.inf
            # Distance of TTS points from PT
            pt = self.filter_labels(["G_temp_sup-Plan_tempo", "PT"])
            dist_pt = dist_calc(self.surf, self.cort, np.where(pt)[0])
            dist_pt[self.ignore] = np.inf
            # Join each point to closer region
            closer_to_hg = (
                np.argmin(np.stack((dist_hg[tts], dist_pt[tts])), axis=0) == 0
            )
            self.labels[np.where(tts)[0][closer_to_hg]] = self.label2num[
                "HG" if self.simplified else "G_temp_sup-G_T_transv"
            ]
            self.labels[np.where(tts)[0][~closer_to_hg]] = self.label2num[
                "PT" if self.simplified else "G_temp_sup-Plan_tempo"
            ]
        else:
            raise ValueError("")

        return self

    def split_stg(self, method="tts_plane"):
        """
        Split STG into middle (mSTG) and posterior (pSTG) halves.

        Parameters
        ----------
        method : {'tts_plane'}, default='tts_plane'
            Method for splitting, currently only support tts_plane.

        Returns
        -------
        self : instance of self
        """
        if self.atlas != 'Destrieux':
            raise ValueError(f'split_stg() only supported for Destrieux atlas.')
        
        if self.is_mangled_stg:
            raise RuntimeError(
                "STG cannot be split as it is already mangled. Try changing order of operations?"
            )
        self.is_mangled_stg = True

        stg = self.filter_labels(["G_temp_sup-Lateral", "STG"])

        if method == "tts_plane":
            # Compute TTS plane
            tts = self.filter_labels(["S_temporal_transverse", "TTS"])
            plane = self.fit_ml_plane_from_line(self.coords[tts])
            # Split STG using the plane
            posterior = (
                np.array([plane.distance_point_signed(p) for p in self.coords[stg]])
                <= 0
            )
            self.labels[np.where(stg)[0][posterior]] = self.label2num[
                "pSTG" if self.simplified else "O_pSTG"
            ]
            self.labels[np.where(stg)[0][~posterior]] = self.label2num[
                "mSTG" if self.simplified else "O_mSTG"
            ]
        else:
            raise ValueError("")

        return self

    def join_ifg(self):
        """
        Join all three subregion of IFG into one.

        Returns:
            self
        """
        if self.is_mangled_ifg:
            raise RuntimeError(
                "IFG cannot be joined as it is already mangled. Try changing order of operations?"
            )
        self.is_mangled_ifg = True

        if self.atlas == 'Destrieux':
            ifg = self.filter_labels(
                [
                    "G_front_inf-Opercular",
                    "G_front_inf-Triangul",
                    "G_front_inf-Orbital",
                    "IFG.opr",
                    "IFG.tri",
                    "IFG.orb",
                ]
            )
        elif self.atlas == 'Desikan-Killiany':
            ifg = self.filter_labels(
                [
                    "parsopercularis",
                    "parstriangularis",
                    "parsorbitalis",
                    "IFG.opr",
                    "IFG.tri",
                    "IFG.orb",
                ]
            )
        else:
            print('No change for coordinate space', self.coordinate_space)
            return self

        self.labels[ifg] = self.label2num["IFG" if self.simplified else "O_IFG"]

        return self

    def reset_overlay(self):
        self.overlay = np.zeros(self.surf[0].shape[0])
        self.alpha = np.ones(self.surf[1].shape[0])
        self.keep_visible = np.ones_like(self.overlay).astype("bool")
        self.keep_visible_cells = np.ones_like(self.alpha).astype("bool")
        return self

    def paint_overlay(self, labels, value=1):
        """
        Paint brain region(s) specified by label(s).

        Returns:
            self
        """
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            if label in self.label2num:
                self.overlay[self.labels==self.label2num[label]] = value
        return self
    
    def interpolate_electrodes_onto_brain(self, coords, values, k, max_dist, roi='all'):
        """
        Use electrode coordinates to interpolate 1-dimensional values corresponding
        to each electrode onto the brain's surface.
        
        Parameters
        ----------
        coords : np.ndarray (elecs, 3)
            3D coordinates of electrodes
        values : np.ndarray (elecs,)
            Value for each electrode
        k : int
            Number of nearest neighbors to consider
        max_dist : float
            Maximum distance outside of which nearest neighbors will be ignored
        roi : list of strings, or string in {'all', 'temporal'}, default='all'
            Regions to allow interpolation over. By default, the entire brain surface
            is allowed. Can also be specified as a list of string labels (drawing from self.label_names)
        
        Notes
        -----
        After running this function, you can use the visualization function ``plot_brain_overlay``
        for a quick matplotlib plot, or you can extract the surface values from the ``self.overlay``
        attribute for plotting with another tool like pysurfer.
        """
        
        if isinstance(roi, str) and roi == 'all':
            roi_list = self.label_names
        elif isinstance(roi, str) and roi == 'temporal':
            if self.atlas != 'Destrieux':
                raise ValueError("roi='temporal' only supported for Destrieux atlas. Must specify list of specific region names")
            if self.simplified:
                roi_list = ['alHG','pmHG','HG','TTS','PT','PP','MTG','ITG','mSTG','pSTG','STG','STS','T.Pole']
            else:
                temporal_regions_nums = [33, 34, 35, 36, 74, 41, 43, 72, 73, 38, 37, 76, 77, 78, 79, 80, 81, 82]
                roi_list = [self.num2label[num] for num in temporal_regions_nums]
        else:
            roi_list = roi
            assert isinstance(roi, list)
            
        roi_list_subset = [x for x in roi_list if x in self.label_names]
        zones_to_include, _, _ = self.zones(roi_list_subset)
        
        # Euclidean distances from each surface vertex to each coordinate
        dists = cdist(self.surf[0], coords)
        sorted_dists = np.sort(dists, axis=-1)[:, :k]
        indices = np.argsort(dists, axis=-1)[:, :k] # get closest k electrodes to each vertex

        # Mask out distances greater than max_dist
        valid_mask = sorted_dists <= max_dist

        # Retrieve the corresponding values using indices
        neighbor_values = values[indices]

        # Mask invalid values
        masked_values = np.where(valid_mask, neighbor_values, np.nan)
        masked_distances = np.where(valid_mask, sorted_dists, np.nan)

        # Compute weights: inverse distance weighting (avoiding division by zero)
        weights = np.where(valid_mask, 1 / (masked_distances + 1e-10), 0)

        # # Compute weighted sum and normalize by total weight per vertex
        weighted_sum = np.nansum(masked_values * weights, axis=1)
        total_weight = np.nansum(weights, axis=1)

        # # Normalize to get final smoothed values
        updated_vertices = np.logical_and(total_weight > 0, zones_to_include)
        total_weight[~updated_vertices] += 1e-10 # this just gets ride of the division by zero warning, but doesn't affect result since these values are turned to nan anyway
        smoothed_values = np.where(updated_vertices, weighted_sum / total_weight, np.nan)

        # update the surface vertices and triangle attributes with the values
        verts = updated_vertices.astype('float')
        trigs = np.zeros(self.n_trigs, dtype=float)
        for i in range(self.n_trigs):
            trigs[i] = np.mean([verts[self.trigs[i, j]] != 0 for j in range(3)])

        self.overlay[updated_vertices] = smoothed_values[updated_vertices]
        
        return self
        

    def mark_overlay(self, verts, value=1, inner_radius=0.8, taper=True):
        """
        Fill circle(s) around target(s).

        Returns
        -------
        self
        """
        if np.isscalar(verts):
            verts = [verts]

        dist = dist_calc(self.surf, self.cort, verts)
        dist[self.ignore] = np.inf

        r1 = dist <= 1 * inner_radius
        r2 = dist <= 2 * inner_radius
        r3 = dist <= 3 * inner_radius

        v1 = value
        v2 = value / 2 if taper else value
        v3 = value / 8 if taper else value

        self.overlay[r3] = np.maximum(self.overlay[r3], v3)
        self.overlay[r2] = np.maximum(self.overlay[r2], v2)
        self.overlay[r1] = np.maximum(self.overlay[r1], v1)

        return self

    def parcellate_overlay(self, merge_func=np.mean):
        """
        Merges overlay values within each parcel for a single hemisphere.

        Parameters
        ----------
        merge_func : callable
            Function to merge values within each parcel.  Should accept a 1D
            NumPy array and return a scalar.
        """
        # Vectorize label to number conversion for efficiency
        label_nums = np.array([self.label2num[label] for label in self.label_names], dtype=self.labels.dtype)
        
        # Vectorize the core logic.
        parcellated_overlay = np.zeros_like(self.overlay) # Create an empty array like self.overlay
        for i, label_num in enumerate(label_nums):
            inds = self.labels == label_num
            if inds.any(): # important check in case a label has no vertices
                parcellated_overlay[inds] = merge_func(self.overlay[inds])
        self.overlay = parcellated_overlay
        return self

    def set_visible(self, labels, min_alpha=0):
        keep_visible, self.alpha, _ = self.zones(labels, min_alpha=min_alpha)
        self.keep_visible = keep_visible > min_alpha
        self.keep_visible_cells = self.alpha > min_alpha
        self.alpha = np.maximum(self.alpha, min_alpha)
        return self
    
    def reset_overlay_except(self, labels):
        keep_visible, self.alpha, _ = self.zones(labels, min_alpha=0)
        self.overlay[~keep_visible] = 0
        return self


class Brain:
    def __init__(
        self,
        surf_type: str = "pial", 
        subject: str = "fsaverage", 
        coordinate_space: str = 'FSAverage',
        atlas=None,
        subject_dir=None
    ):
        """
        Brain representation containing a left and right hemisphere. Can be used for plotting,
        distance calculations, etc.

        Parameters
        ----------
        surf_type : str, default='pial'
            Cortical surface type, either 'pial' or 'inflated' or another if the corresponding
            files can be found.
        subject : str, default='fsaverage'
            Subject to use, must be a directory within ``subject_dir``
        coordinate_space : str, default='FSAverage'
            Coordinate space of brain vertices. Must be 'FSAverage' or 'MNI152'
        atlas : str, default=None
            Atlas for brain parcellation. Defaults to 'Destrieux' for coordinate_space='FSAverage'
            and 'Desikan-Killiany' for 'MNI152'. Can also be an annotation file name given by
            ``{subject_dir}/{subject}/label/?h.{atlas}.annot``
        subject_dir : str/path-like, defaults to SUBJECT_DIR environment variable, or the current directory
            if that does not exist.
            Path containing the subject's folder.

        Examples
        --------
        >>> from naplib.localization import Brain
        >>> from naplib.visualization import plot_brain_elecs
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> brain = Brain('pial', subject_dir='path/to/freesurfer/subjects/').split_stg().join_ifg()
        >>> coords = np.array([[-47.281147  ,  17.026093  , -21.833099  ],
                               [-48.273964  ,  16.155487  , -20.162935  ]])
        >>> isleft = np.array([True, True])
        >>> annotations = brain.annotate_coords(coords, isleft)
        array(['mSTG','mSTG'])
        >>> dist_from_HG = brain.distance_from_region(coords, isleft, region='pmHG', metric='surf')
        array([52.67211969 50.86446306])
        >>> # plot electrodes on brain with matplotlib
        >>> fig, axes = plot_brain_elecs(brain, coords, isleft, values=dist_from_HG, hemi='lh', view='lateral')
        >>> plt.show()
        >>> # plot electrodes on brain in interactive 3D figure
        >>> fig, _ = plot_brain_elecs(brain, coords, isleft, values=dist_from_HG, backend='plotly')
        >>> fig.write_html("interactive_brain_plot.html") # save as an interactive html figure
        >>> fig.show()

        """
        if surf_type not in SURF_TYPES:
            raise ValueError(f"Argument `surf_type` should be in {SURF_TYPES}.")

        self.surf_type = surf_type
        self.subject = subject

        self.lh = Hemisphere("lh", surf_type, subject, coordinate_space, atlas, subject_dir=subject_dir)
        self.rh = Hemisphere("rh", surf_type, subject, coordinate_space, atlas, subject_dir=subject_dir)

    @property
    def num2label(self):
        return self.lh.num2label

    @property
    def label2num(self):
        return self.lh.label2num

    @property
    def label_names(self):
        return list(set(self.lh.label_names + self.rh.label_names))

    def load_labels(self):
        """
        Load Destrieux labels of each vertex from annotation files.
        """
        self.lh.load_labels()
        self.rh.load_labels()
        return self

    def simplify_labels(self):
        """
        Simplify Destrieux labels into shortforms.
        """
        self.lh.simplify_labels()
        self.rh.simplify_labels()
        return self

    def split_hg(self, method="midpoint"):
        """
        Split HG vertices into posteromedial (pmHG) and anterolateral (alHG) halves.

        Arguments:
            method, str: How to split the halves. One of 'midpoint', 'endpoint' or 'median'.
        """
        self.lh.split_hg(method)
        self.rh.split_hg(method)
        return self

    def split_stg(self, method="tts_plane"):
        """
        Split STG into middle (mSTG) and posterior (pSTG) halves.
        """
        self.lh.split_stg(method)
        self.rh.split_stg(method)
        return self

    def remove_tts(self, method="split"):
        """
        Convert TTS labels into either HG or PT ones.
        """
        self.lh.remove_tts(method)
        self.rh.remove_tts(method)
        return self

    def join_ifg(self):
        """
        Join all three subregion of IFG into one.
        """
        self.lh.join_ifg()
        self.rh.join_ifg()
        return self

    def annotate(self, verts, is_left, is_surf=None, text=True):
        """
        Get labels for vertices of the surface.

        Parameters
        ----------
        verts : np.ndarray
            Array of vertices.
        isleft : np.ndarray
            Boolean array whether each vertex belongs to the left hemisphere.
        distance_cutoff : float, default=10
            Electrodes further than this distance (in mm) from the cortical surface will be labeled as "Other"
        is_surf : boolean np.ndarray
            Array of the same shape as the number of vertices in the surface (e.g. len(self.lh.surf[0])) indicating
            whether those points should be included as surface options. If an electrode is closest to a point
            with a False indicator in this array, then it will get None as its label.
        text : bool, default=True
            Whether to return labels as string names, or integer labels.

        Returns
        -------
        labels : np.ndarray
            Array of labels, either as strings or ints.

        """
        labels = np.zeros(len(verts), dtype=int)
        labels[is_left] = self.lh.labels[verts[is_left]]
        labels[~is_left] = self.rh.labels[verts[~is_left]]
        labels[verts <= 1] = 0
        if is_surf is not None:
            labels[~is_surf] = 0
        if text:
            labels = np.array([self.lh.num2label[label] if is_left[i] else self.rh.num2label[label]
             for i,label in enumerate(labels)])
        return labels

    def annotate_coords(
        self, coords, isleft=None, distance_cutoff=10, is_surf=None, text=True, get_dists=False,
    ):
        """
        Get labels (like pmHG, IFG, etc) for coordinates. Note, the coordinates should match the
        `surf_type` of this brain, otherwise finding nearest surface points to each coordinate in order
        to label it may be inaccurate.

        Parameters
        ----------
        coords : np.ndarray
            Array of coordinates, shape (num_elecs, 3).
        isleft : np.ndarray (elecs,), optional
            If provided, specifies a boolean which is True for each electrode that is in the left hemisphere.
            If not given, this will be inferred from the first dimension of the coords (negative is left).
        distance_cutoff : float, default=10
            Electrodes further than this distance (in mm) from the cortical surface will be labeled as None
        is_surf : boolean np.ndarray
            Array of the same shape as the number of vertices in the surface (e.g. len(self.lh.surf[0])) indicating
            whether those points should be included as surface options. If an electrode is closest to a point
            with a False indicator in this array, then it will get None as its label.
        text : bool, default=True
            Whether to return labels as string names, or integer labels.
        get_dists : bool, default=False
            Whether to return distances for each electrode to the nearest vertex.

        Returns
        -------
        labels : np.ndarray
            Array of labels, either as strings or ints.
        dists : np.ndarray, optional
            Array of minimum distances as floats

        """
        if isleft is None:
            isleft = coords[:,0] < 0

        verts, dists = get_nearest_vert_index(
            coords, isleft, self.lh.surf, self.rh.surf, verbose=False
        )
        labels = self.annotate(verts, isleft, is_surf=is_surf, text=text)
        labels = np.asarray(
            [
                lab if dist < distance_cutoff else None
                for lab, dist in zip(labels, dists)
            ]
        )
        if get_dists:
            return labels, dists
        else:
            return labels

    def distance_from_region(self, coords, isleft=None, region="pmHG", metric="surf"):
        """
        Get distance from a certain region for each electrode's coordinates. Can compute
        distance along the cortical surface or as euclidean distance. For proper results, assuming
        coordinates are in pial space, the brain must also be in pial space.

        Parameters
        ----------
        coords : np.ndarray
            Array of coordinates in pial space for this brain's subject_id, shape (num_elecs, 3).
        isleft : np.ndarray (elecs,), optional
            If provided, specifies a boolean which is True for each electrode that is in the left hemisphere.
            If not given, this will be inferred from the first dimension of the coords (negative is left).
        region : str, default='pmHG'
            Anatomical label. Must exist in the labels for the brain.
        metric : {'surf','euclidean'}, default='surf'
            Either surf, for distance along cortical surface, or euclidean, for euclidean distance.

        Returns
        -------
        distances : np.ndarray
            Array of distances, in mm.
        """
        if isleft is None:
            isleft = coords[:,0] < 0

        region_label_num = self.label2num[region]

        if region_label_num not in self.lh.labels:
            raise ValueError(
                "Region not found in existing labels. One possible issue is that you have not yet called"
                " brain.split_hg(), or a similar method. For example, Te1.1 is only"
                " available after calling brain.split_hg(method='te1x')"
            )

        surf_lh = self.lh.surf
        surf_rh = self.rh.surf

        # see which vertices correspond to this region in each hemi
        which_verts_this_region_lh = self.lh.labels == region_label_num
        which_verts_this_region_rh = self.rh.labels == region_label_num

        # find the center of this region
        region_center_lh = surf_lh[0][which_verts_this_region_lh].mean(0, keepdims=True)
        region_center_rh = surf_rh[0][which_verts_this_region_rh].mean(0, keepdims=True)

        if metric == "surf":
            # get the closest valid vertex to this center point
            closest_surface_vert_to_region_center_lh = np.argmin(
                np.square(surf_lh[0] - region_center_lh.squeeze()).sum(1)
            )
            closest_surface_vert_to_region_center_rh = np.argmin(
                np.square(surf_rh[0] - region_center_rh.squeeze()).sum(1)
            )

            # get distance from every vertex on the surface to this vertex
            dist_lh = dist_calc(
                surf_lh, self.lh.cort, closest_surface_vert_to_region_center_lh
            )
            dist_rh = dist_calc(
                surf_rh, self.rh.cort, closest_surface_vert_to_region_center_rh
            )

            # get approximate vertex for every coordinate
            nearest_verts, _ = get_nearest_vert_index(
                coords, isleft, surf_lh, surf_rh, verbose=False
            )

            # get distance for each of these vertices from the region center, which was already calculated
            distances_by_elec = []
            for i in range(len(coords)):
                if isleft[i]:
                    distances_by_elec.append(dist_lh[nearest_verts[i]])
                else:
                    distances_by_elec.append(dist_rh[nearest_verts[i]])

        elif metric == "euclidean":
            dist_lh = cdist(region_center_lh, coords).squeeze()
            dist_rh = cdist(region_center_rh, coords).squeeze()
            distances_by_elec = []
            for i in range(len(coords)):
                if isleft[i]:
                    distances_by_elec.append(dist_lh[i])
                else:
                    distances_by_elec.append(dist_rh[i])

        else:
            raise ValueError(f"metric must be surf or euclidean but got {metric}")

        return np.asarray(distances_by_elec)

    def reset_overlay(self):
        self.lh.reset_overlay()
        self.rh.reset_overlay()
        return self

    def paint_overlay(self, labels, value=1):
        """
        Paint brain region(s) specified by label(s).

        Parameters
        ----------
        labels : str | list[str]
            Region or regions to paint an overlay.
        value : float, default=1
            Value to paint the region overlay with.

        Returns
        -------
        self : an instance of self
        """
        self.lh.paint_overlay(labels, value)
        self.rh.paint_overlay(labels, value)
        return self

    def mark_overlay(self, verts, isleft, value=1, inner_radius=0.8, taper=True):
        """
        Fill circle(s) around target(s).

        Parameters
        ----------
        verts : np.ndarray
            Vertices to mark.
        isleft : np.ndarray of booleans
            Indicator of same shape as verts for whether they are in the left hemisphere.
        value : float, default=1
            Value to mark with.
        inner_radius : float, default=0.8
            Radius of circle to mark around each vertex.
        taper : bool, default=True
            Whether to taper the circular mark.

        Returns
        -------
        self : instance of self
        """
        self.lh.mark_overlay(verts[isleft], value, inner_radius, taper)
        self.rh.mark_overlay(verts[~isleft], value, inner_radius, taper)
        return self

    def set_visible(self, labels, min_alpha=0):
        """
        Set certain regions as visible with a float label, and the rest will be invisible.

        Parameters
        ----------
        labels : str | list[str]
            Label(s) to set as visible.
        min_alpha : float, default=0


        Returns
        -------
        self : instance of self
        """
        self.lh.set_visible(labels, min_alpha)
        self.rh.set_visible(labels, min_alpha)
        return self
    
    def reset_overlay_except(self, labels):
        """
        Keep certain regions and the rest as colorless.

        Parameters
        ----------
        labels : str | list[str]
            Label(s) to set as visible.

        Returns
        -------
        self : instance of self
        """
        self.lh.reset_overlay_except(labels)
        self.rh.reset_overlay_except(labels)
        return self
    
    def interpolate_electrodes_onto_brain(self, coords, values, isleft=None, k=10, max_dist=10, roi='all', reset_overlay_first=True):
        """
        Use electrode coordinates to interpolate 1-dimensional values corresponding
        to each electrode onto the brain's surface.
        
        Parameters
        ----------
        coords : np.ndarray (elecs, 3)
            3D coordinates of electrodes
        values : np.ndarray (elecs,)
            Value for each electrode
        isleft : np.ndarray (elecs,), optional
            If provided, specifies a boolean which is True for each electrode that is in the left hemisphere.
            If not given, this will be inferred from the first dimension of the coords (negative is left).
        k : int, default=10
            Number of nearest neighbors to consider
        max_dist : float, default=10
            Maximum distance (in mm) outside of which nearest neighbors will be ignored
        roi : list of strings, or string in {'all', 'temporal'}, default='all'
            Regions to allow interpolation over. By default, the entire brain surface
            is allowed. Can also be specified as a list of string labels (drawing from self.lh.label_names)
        reset_overlay_first : bool, default=True
            If True (default), reset the overlay before creating a new overlay
        
        Notes
        -----
        After running this function, you can use the visualization function ``plot_brain_overlay``
        for a quick matplotlib plot, or you can extract the surface values from the ``self.lh.overlay``
        and ``self.rh.overlay`` attributes, etc, for plotting with another tool like pysurfer or plotly.
        """
        if reset_overlay_first:
            self.reset_overlay()
        if isleft is None:
            isleft = coords[:,0] < 0
        self.lh.interpolate_electrodes_onto_brain(coords[isleft], values[isleft], k=k, max_dist=max_dist, roi=roi)
        self.rh.interpolate_electrodes_onto_brain(coords[~isleft], values[~isleft], k=k, max_dist=max_dist, roi=roi)
        return self

    def parcellate_overlay(self, merge_func=np.mean):
        """Merges brain overlay values within each atlas parcel.

        This method applies a merging function to the overlay values within each
        anatomical parcel defined by an atlas.  It is typically used after
        interpolating electrode data onto the brain surface
        (e.g., via `brain.interpolate_electrodes_onto_brain()`) to summarize
        the data within each parcel.

        Parameters
        ----------
        merge_func : callable, optional
            The function used to combine the overlay values within each parcel.
            The function should accept an array-like object of values and return a
            single value.  Common examples include `numpy.mean` (default),
            `numpy.median`, and `numpy.max`.

        Returns
        -------
        self : instance of self
            Returns the instance itself, with the overlay data parcellated.
        """
        self.lh.parcellate_overlay(merge_func)
        self.rh.parcellate_overlay(merge_func)
        return self


def get_nearest_vert_index(coords, isleft, surf_lh, surf_rh, verbose=False):
    vert_indices = []
    min_dists = []
    # loop through coordinates and update stat for each node that this electrode coordinate is close enough to
    for i, coord in enumerate(coords):
        if isleft[i]:
            dists = np.sqrt(np.square(surf_lh[0] - coord).sum(1))
        else:
            dists = np.sqrt(np.square(surf_rh[0] - coord).sum(1))
        min_dists.append(dists.min())
        if verbose:
            print(min_dists[-1])
        vert_indices.append(dists.argmin())

    return np.asarray(vert_indices), np.asarray(min_dists)


def find_closest_vertices(surface_coords, point_coords):
    """Return the vertices on a surface mesh closest to some given coordinates.

    The distance metric used is Euclidian distance.

    Parameters
    ----------
    surface_coords : numpy array
        Array of coordinates on a surface mesh
    point_coords : numpy array
        Array of coordinates to map to vertices

    Returns
    -------
    closest_vertices : numpy array
        Array of mesh vertex ids

    """
    point_coords = np.atleast_2d(point_coords)
    dists = cdist(surface_coords, point_coords)
    return np.argmin(dists, axis=0), np.min(dists, axis=0)
