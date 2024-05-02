"""
Functions for localizing and plotting intracranial electrodes on the freesurfer average brain.
"""

import os
import warnings
from os.path import join as pjoin

import numpy as np
from nibabel.freesurfer.io import read_geometry, read_label, read_morph_data
from scipy.spatial.distance import cdist
from skspatial.objects import Line, Plane
from hdf5storage import loadmat

from naplib.utils import dist_calc, load_freesurfer_label
from naplib import logger

warnings.filterwarnings("ignore", message="nopython", append=True)

HEMIS = ("lh", "rh")
SURF_TYPES = ("pial", "inflated")


num2region = {
    # Unknown
    0: "Unknown",
    # Destrieux labels
    1: "G_and_S_frontomargin",
    2: "G_and_S_occipital_inf",
    3: "G_and_S_paracentral",
    4: "G_and_S_subcentral",
    5: "G_and_S_transv_frontopol",
    6: "G_and_S_cingul-Ant",
    7: "G_and_S_cingul-Mid-Ant",
    8: "G_and_S_cingul-Mid-Post",
    9: "G_cingul-Post-dorsal",
    10: "G_cingul-Post-ventral",
    11: "G_cuneus",
    12: "G_front_inf-Opercular",
    13: "G_front_inf-Orbital",
    14: "G_front_inf-Triangul",
    15: "G_front_middle",
    16: "G_front_sup",
    17: "G_Ins_lg_and_S_cent_ins",
    18: "G_insular_short",
    19: "G_occipital_middle",
    20: "G_occipital_sup",
    21: "G_oc-temp_lat-fusifor",
    22: "G_oc-temp_med-Lingual",
    23: "G_oc-temp_med-Parahip",
    24: "G_orbital",
    25: "G_pariet_inf-Angular",
    26: "G_pariet_inf-Supramar",
    27: "G_parietal_sup",
    28: "G_postcentral",
    29: "G_precentral",
    30: "G_precuneus",
    31: "G_rectus",
    32: "G_subcallosal",
    33: "G_temp_sup-G_T_transv",
    34: "G_temp_sup-Lateral",
    35: "G_temp_sup-Plan_polar",
    36: "G_temp_sup-Plan_tempo",
    37: "G_temporal_inf",
    38: "G_temporal_middle",
    39: "Lat_Fis-ant-Horizont",
    40: "Lat_Fis-ant-Vertical",
    41: "Lat_Fis-post",
    42: "Pole_occipital",
    43: "Pole_temporal",
    44: "S_calcarine",
    45: "S_central",
    46: "S_cingul-Marginalis",
    47: "S_circular_insula_ant",
    48: "S_circular_insula_inf",
    49: "S_circular_insula_sup",
    50: "S_collat_transv_ant",
    51: "S_collat_transv_post",
    52: "S_front_inf",
    53: "S_front_middle",
    54: "S_front_sup",
    55: "S_interm_prim-Jensen",
    56: "S_intrapariet_and_P_trans",
    57: "S_oc_middle_and_Lunatus",
    58: "S_oc_sup_and_transversal",
    59: "S_occipital_ant",
    60: "S_oc-temp_lat",
    61: "S_oc-temp_med_and_Lingual",
    62: "S_orbital_lateral",
    63: "S_orbital_med-olfact",
    64: "S_orbital-H_Shaped",
    65: "S_parieto_occipital",
    66: "S_pericallosal",
    67: "S_postcentral",
    68: "S_precentral-inf-part",
    69: "S_precentral-sup-part",
    70: "S_suborbital",
    71: "S_subparietal",
    72: "S_temporal_inf",
    73: "S_temporal_sup",
    74: "S_temporal_transverse",
    # My custom labels
    75: "O_pmHG",
    76: "O_alHG",
    77: "O_Te10",
    78: "O_Te11",
    79: "O_Te12",
    80: "O_mSTG",
    81: "O_pSTG",
    82: "O_IFG",
}
region2num = {v: k for k, v in num2region.items()}

num2region_mni = {
    0: 'unknown',
    1: 'bankssts',
    2: 'caudalanteriorcingulate',
    3: 'caudalmiddlefrontal',
    4: 'corpuscallosum',
    5: 'cuneus',
    6: 'entorhinal',
    7: 'fusiform',
    8: 'inferiorparietal',
    9: 'inferiortemporal',
    10: 'isthmuscingulate',
    11: 'lateraloccipital',
    12: 'lateralorbitofrontal',
    13: 'lingual',
    14: 'medialorbitofrontal',
    15: 'middletemporal',
    16: 'parahippocampal',
    17: 'paracentral',
    18: 'parsopercularis',
    19: 'parsorbitalis',
    20: 'parstriangularis',
    21: 'pericalcarine',
    22: 'postcentral',
    23: 'posteriorcingulate',
    24: 'precentral',
    25: 'precuneus',
    26: 'rostralanteriorcingulate',
    27: 'rostralmiddlefrontal',
    28: 'superiorfrontal',
    29: 'superiorparietal',
    30: 'superiortemporal',
    31: 'supramarginal',
    32: 'frontalpole',
    33: 'temporalpole',
    34: 'transversetemporal',
    35: 'insula',
    36: 'cMTG',
    37: 'mMTG',
    38: 'rMTG',
    39: 'cSTG',
    40: 'mSTG',
    41: 'rSTG',
    # My custom labels
    42: "O_IFG",
}
region2num_mni = {v: k for k, v in num2region_mni.items()}


class Hemisphere:
    def __init__(
        self,
        hemi: str,
        surf_type: str = "pial",
        subject: str = "fsaverage",
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
        subject_dir : str/path-like, defaults to SUBJECT_DIR environment variable, or the current directory
            if that does not exist.
            Path containing the subject's folder.
        """
        if hemi not in HEMIS:
            raise ValueError(f"Argument `hemi` should be in {HEMIS}.")
        if surf_type not in SURF_TYPES:
            raise ValueError(f"Argument `surf_type` should be in {SURF_TYPES}.")

        self.hemi = hemi
        self.surf_type = surf_type
        self.subject = subject

        if subject_dir is None:
            subject_dir = os.environ.get("SUBJECTS_DIR", "./")

        self.subject_dir = subject_dir
        
        self.atlas = 'FSAverage'

        if os.path.exists(self.surf_file(f"{hemi}.{surf_type}")):
            self.surf = read_geometry(self.surf_file(f"{hemi}.{surf_type}"))
        else:
            # try to find .mat file
            surf_ = loadmat(self.surf_file(f"{hemi}_pial.mat"))
            coords, faces = surf_['coords'], surf_['faces']
            faces -= 1 # make faces zero-indexed
            self.surf = (coords, faces)
            self.atlas = 'MNI152'
        
        if self.atlas == 'FSAverage':
            self.surf_pial = read_geometry(self.surf_file(f"{hemi}.pial"))
        else:
            # try to find .mat file
            surf_ = loadmat(self.surf_file(f"{hemi}_pial.mat"))
            coords, faces = surf_['coords'], surf_['faces']
            faces -= 1 # make faces zero-indexed
            self.surf_pial = (coords, faces)
        
        try:
            self.cort = np.sort(read_label(self.label_file(f"{hemi}.cortex.label")))
        except Exception as e:
            logger.warning(f'No {hemi}.cortext.label file found. Assuming the entire surface is cortex.')
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
        annot_file = self.label_file(f"{self.hemi}.aparc.a2009s.annot")
        annot_file_mni = self.label_file(f"FSL_MNI152.{self.hemi}.aparc.split_STG_MTG.annot")
        if self.atlas == 'FSAverage':
            for ind, reg in num2region.items():
                if reg.startswith("O"):
                    continue
                self.labels[load_freesurfer_label(annot_file, reg)] = ind
        elif self.atlas == 'MNI152':
            for ind, reg in num2region_mni.items():
                if reg.startswith("O"):
                    continue
                self.labels[load_freesurfer_label(annot_file_mni, reg)] = ind
        else:
            raise ValueError('Bad atlas')
        self.labels[self.ignore] = 0
        if self.atlas == 'FSAverage':
            self.num2label = num2region
            self.label2num = {v: k for k, v in self.num2label.items()}
        elif self.atlas == 'MNI152':
            self.num2label = num2region_mni
            self.label2num = {v: k for k, v in self.num2label.items()}
        else:
            raise ValueError('Bad atlas')
            
        self.simplified = False

        self.is_mangled_hg = False
        self.is_mangled_tts = False
        self.is_mangled_stg = False
        self.is_mangled_ifg = False

        return self

    def simplify_labels(self):
        """
        Simplify Destrieux labels into shortforms.

        Returns
        -------
        self : instance of self
        """
        if self.atlas == 'FSAverage':
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
            

        elif self.atlas == 'MNI152':
            d1 = {k: [k] for k in region2num_mni.keys() if k not in ['O_IFG','parsopercularis','parstriangularis','parsorbitalis']}
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

        if self.atlas == 'MNI152':
            raise ValueError(f'split_hg() is not supported for MNI atlas.')
            
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
        if self.atlas == 'MNI152':
            raise ValueError(f'remove_tts() is not supported for MNI atlas.')
        
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
        if self.atlas == 'MNI152':
            raise ValueError(f'split_stg() is not supported for MNI atlas.')
        
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

        if self.atlas == 'FSAverage':
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
        else: # MNI152
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

        self.labels[ifg] = self.label2num["IFG" if self.simplified else "O_IFG"]

        return self

    def reset_overlay(self):
        self.overlay = np.zeros(self.surf[0].shape[0])
        self.alpha = np.ones(self.surf[1].shape[0])
        self.keep_visible = np.ones_like(self.overlay).astype("bool")
        self.keep_visible_cells = np.ones_like(self.alpha).astype("bool")
        self.has_overlay = np.zeros_like(self.overlay).astype("bool")
        self.has_overlay_cells = np.zeros_like(self.alpha).astype("bool")
        return self

    def paint_overlay(self, labels, value=1):
        """
        Paint brain region(s) specified by label(s).

        Returns:
            self
        """
        verts, add_overlay, _ = self.zones(labels)
        self.overlay[verts] = value
        self.has_overlay[verts == 1] = True
        self.has_overlay_cells[add_overlay == 1] = True
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

    def set_visible(self, labels, min_alpha=0):
        keep_visible, self.alpha, _ = self.zones(labels, min_alpha=min_alpha)
        self.keep_visible = keep_visible > min_alpha
        self.keep_visible_cells = self.alpha > min_alpha
        self.alpha = np.maximum(self.alpha, min_alpha)
        return self


class Brain:
    def __init__(
        self, surf_type: str = "pial", subject: str = "fsaverage", subject_dir=None
    ):
        """
        Brain representation containing a left and right hemisphere. Can be used for plotting,
        distance calculations, etc.

        Parameters
        ----------
        hemi : str
            Either 'lh' or 'rh'.
        surf_type : str, default='pial'
            Cortical surface type, either 'pial' or 'inflated' or another if the corresponding
            files can be found.
        subject : str, default='fsaverage'
            Subject to use, must be a directory within ``subject_dir``
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

        self.lh = Hemisphere("lh", surf_type, subject, subject_dir=subject_dir)
        self.rh = Hemisphere("rh", surf_type, subject, subject_dir=subject_dir)

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
            labels = np.array([self.num2label[label] for label in labels])
        return labels

    def annotate_coords(
        self, coords, isleft, distance_cutoff=10, is_surf=None, text=True
    ):
        """
        Get labels (like pmHG, IFG, etc) for coordinates. Note, the coordinates should match the
        `surf_type` of this brain, otherwise finding nearest surface points to each coordinate in order
        to label it may be inaccurate.

        Parameters
        ----------
        coords : np.ndarray
            Array of coordinates, shape (num_elecs, 3).
        isleft : np.ndarray
            Boolean array whether each electrode belongs to the left hemisphere, shape (num_elecs,).
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
        return labels

    def distance_from_region(self, coords, isleft, region="pmHG", metric="surf"):
        """
        Get distance from a certain region for each electrode's coordinates. Can compute
        distance along the cortical surface or as euclidean distance. For proper results, assuming
        coordinates are in pial space, the brain must also be in pial space.

        Parameters
        ----------
        coords : np.ndarray
            Array of coordinates in pial space for this brain's subject_id, shape (num_elecs, 3).
        isleft : np.ndarray
            Boolean array whether each electrode belongs to the left hemisphere, shape (num_elecs,).
        region : str, default='pmHG'
            Anatomical label. Must exist in the labels for the brain.
        metric : {'surf','euclidean'}, default='surf'
            Either surf, for distance along cortical surface, or euclidean, for euclidean distance.

        Returns
        -------
        distances : np.ndarray
            Array of distances, in mm.
        """

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
        Set certain regions as visible with a float label.

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

