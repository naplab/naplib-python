"""
Functions for localizing and plotting intracranial electrodes on the freesurfer average brain.
"""

import os
import warnings
from os.path import join as pjoin

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nibabel.freesurfer.io import read_geometry, read_label, read_morph_data
from scipy.spatial.distance import cdist
from skspatial.objects import Line, Plane

from naplib.utils import dist_calc, load_freesurfer_label, surfdist_viz

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


ROI = (
    # 'G_and_S_subcentral',
    "G_front_inf-Opercular",
    "G_front_inf-Orbital",
    "G_front_inf-Triangul",
    # 'G_Ins_lg_and_S_cent_ins',
    # 'G_insular_short',
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo",
    # 'G_temporal_inf',
    "G_temporal_middle",
    # 'Pole_temporal',
    # 'S_circular_insula_ant',
    # 'S_circular_insula_inf',
    # 'S_circular_insula_sup',
    # 'S_front_inf',
    # 'S_temporal_inf',
    # 'S_temporal_sup',
    "S_temporal_transverse",
    "O_pmHG",
    "O_alHG",
    "O_Te10",
    "O_Te11",
    "O_Te12",
    "O_mSTG",
    "O_pSTG",
    "O_IFG",
)
ROI_SIMPLE = (
    "HG",
    "pmHG",
    "alHG",
    "Te1.0",
    "Te1.1",
    "Te1.2",
    "TTS",
    "PT",
    "PP",
    "STG",
    "MTG",
    "mSTG",
    "pSTG",
    "IFG",
    "IFG.opr",
    "IFG.tri",
    "IFG.orb",
)


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

        self.surf = read_geometry(self.surf_file(f"{hemi}.{surf_type}"))
        self.cort = np.sort(read_label(self.label_file(f"{hemi}.cortex.label")))
        self.sulc = read_morph_data(self.surf_file(f"{hemi}.sulc"))

        self.surf_pial = read_geometry(self.surf_file(f"{hemi}.pial"))

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

    def view(self, mode: str = "lateral", backend: str = "mpl"):
        """
        Appropriate azimuth for displaying the specified hemisphere in specified view.

        Arguments
        ---------
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
                eye = dict(x=-1, y=0, z=0) if self.hemi == "lh" else dict(x=1, y=0, z=0)
                center = dict(x=0, y=0, z=0)
                return eye, center
            else:
                return (0, 180) if self.hemi == "lh" else (0, 0)

        elif mode == "medial":
            if backend == "plotly":
                eye = dict(x=1, y=0, z=0) if self.hemi == "lh" else dict(x=-1, y=0, z=0)
                center = dict(x=0, y=0, z=0)
                return eye, center
            else:
                return (0, 0) if self.hemi == "lh" else (0, 180)
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
                eye = (
                    dict(x=-1, y=0.1, z=0.1)
                    if self.hemi == "lh"
                    else dict(x=1, y=0.1, z=0.1)
                )
                center = dict(x=0, y=0, z=0)
                return eye, center
            else:
                return (20, 160) if self.hemi == "lh" else (40, 20)
        else:
            raise ValueError(f"Unknown `mode`: {mode}.")

    def load_labels(self):
        """
        Load Destrieux labels of each vertex from annotation files.

        Returns
        -------
        self : instance of self
        """
        self.ignore = np.zeros(self.n_verts, dtype=bool)
        annot_file = self.label_file(f"{self.hemi}.aparc.a2005s.annot")
        for reg in ("Unknown", "Medial_wall"):
            self.ignore[load_freesurfer_label(annot_file, reg)] = True

        self.labels = np.zeros(self.n_verts, dtype=int)
        annot_file = self.label_file(f"{self.hemi}.aparc.a2009s.annot")
        for ind, reg in num2region.items():
            if reg.startswith("O"):
                continue
            self.labels[load_freesurfer_label(annot_file, reg)] = ind
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
        Simplify Destrieux labels into shortforms.

        Returns
        -------
        self : instance of self
        """
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
        }
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

    def filter_labels(self, labels=None):
        """
        Returns mask of vertices that are within the union of `labels`.
        """
        if isinstance(labels, str):
            labels = (labels,)
        elif labels is None:
            labels = ROI_SIMPLE if self.simplified else ROI

        mask = np.zeros(self.n_verts, dtype=bool)
        for label in labels:
            if label in self.label2num:
                mask[self.labels == self.label2num[label]] = True

        return mask

    def zones(self, labels=None, min_alpha=0):
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
        elif labels is None:
            labels = ROI_SIMPLE if self.simplified else ROI

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

    def plot(self, cmap="coolwarm", ax=None, denorm=False, view="best"):
        surfdist_viz(
            *self.surf,
            self.overlay,
            *self.view(view),
            cmap=cmap(self.overlay.max()) if denorm else cmap,
            threshold=0.25,
            alpha=self.alpha,
            bg_map=self.sulc,
            bg_on_stat=True,
            ax=ax,
        )
        ax.axes.set_axis_off()
        ax.grid(False)


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
        >>> brain = Brain('pial', subject_dir='path/to/freesurfer/subjects/').split_stg().join_ifg()
        >>> annotations = brain.

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
        self.lh.split_stg()
        self.rh.split_stg()
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

    def plot_brain_overlay(
        self, cmap="coolwarm", ax=None, denorm=False, view="best", **kwargs
    ):
        """
        Plot brain overlay on the 3D cortical surface. If certain regions have been set as visible using
        self.set_visible, only those regions will be shown.

        Parameters
        ----------
        cmap : str, default='coolwarm'
            Colormap to use.
        ax : list | tuple of matplotlib Axes
            2 Axes to plot the left and right hemispheres with.
        denorm : bool, default=False
            Whether to center the overlay labels around 0 or not before sending to the colormap.
        view : str, default='best'
            Which view to plot for each hemisphere.
        **kwargs : kwargs
            Any other kwargs to pass to matplotlib.pyplot.figure

        """
        fig = plt.figure(**kwargs)
        if ax is None:
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        else:
            ax1, ax2 = ax

        self.lh.plot(cmap, ax1, denorm, view=view)
        self.rh.plot(cmap, ax2, denorm, view=view)

    def plot_brain_elecs(
        self,
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
        Plot electrodes on the brain using a simple matplotlib backend.

        Due to the limitation of matplotlib being unable to render 3D surfaces
        in order as they would truly be seen by the camera angle, electrodes
        which are behind the cortical surface will still be visible as if
        they were in front of it.

        Parameters
        ----------
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
        elec_size : int, default=4
            Size of the markers representing electrodes.
        cortex : {'classic','high_contrast','mid_contrast','low_contrast','bone'}, default='classic'
            How to map the sulci to greyscale. 'classic' will leave sulci untouched, which may be
            better for plotting the pial surface, but 'high_contrast' will enhance the contrast between
            gyri and sulci, which may be better for inflated surfaces.
        cmap : str, default='cool'
            Colormap for electrode values if values are provided.
        alpha : float | np.ndarray
            Opacity of the electrodes. Either a single float or an array of same length as number of electrodes.
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

        If plots still don't show in the notebook, you may need to enable certain ipywidgets for displaying plotly in notebooks.
        Regardless, it should still be possible to save the the figure as either HTML or
        static image file. See `plotly examples documentation <https://plotly.com/python/interactive-html-export/>`_ for
        details on saving figures.

        """
        if hemi == "both":
            surfs = {"lh": self.lh.surf, "rh": self.rh.surf}
            sulci = {"lh": self.lh.sulc, "rh": self.rh.sulc}
        elif hemi == "lh":
            surfs = {"lh": self.lh.surf}
            sulci = {"lh": self.lh.sulc}
        elif hemi == "rh":
            surfs = {"rh": self.rh.surf}
            sulci = {"rh": self.rh.sulc}
        else:
            raise ValueError(f"hemi must be either both, lh, or rh, but got {hemi}")

        if isleft is None:
            isleft = elecs[:, 0] < 0

        if backend not in ["mpl", "plotly"]:
            raise ValueError(f"backend must be either mpl or plotly but got {backend}")

        if snap_to_surface is None:
            if self.surf_type == "pial":
                snap_to_surface = False
            else:
                snap_to_surface = True

        if brain_alpha is None:
            if self.surf_type == "pial":
                brain_alpha = 0.3
            else:
                if backend == "plotly":
                    brain_alpha = 1
                else:
                    brain_alpha = 0.45

        fig, axes = self._plot_brain_elecs_standalone(
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
        self,
        surfs,
        sulci,
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
            classic=(dict(colormap="Greys", vmin=-1, vmax=2), None),
            high_contrast=(dict(colormap="Greys", vmin=-0.2, vmax=1.3), None),
            mid_contrast=(dict(colormap="Greys", vmin=-1.3, vmax=1.3), np.tanh),
            low_contrast=(dict(colormap="Greys", vmin=-4, vmax=4), None),
            grey_binary=(
                dict(colormap="Greys", vmin=-0.9, vmax=2),
                lambda arr: np.where(arr < 0.5, np.tanh(arr - 0.2), np.tanh(arr + 0.2)),
            ),
            bone=(dict(colormap="bone", vmin=-0.2, vmax=2), None),
        )

        assert isinstance(surfs, dict)
        assert isinstance(sulci, dict)

        if cortex not in colormap_map:
            raise ValueError(
                f"Invalid cortex. Must be one of {'classic','high_contrast','low_contrast','bone_r'} but got {cortex}"
            )

        sulci_cmap_kwargs, sulci_cmap_nonlinearity = colormap_map[cortex]

        # this must be defined within this scope so it can use the graph_objects import
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

        # this must be defined within this scope so it can use the graph_objects import
        def _plotly_scatter3d(coords, elec_colors, name=""):
            marker = go.scatter3d.Marker(color=elec_colors)
            scatter = go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=marker,
                name=name,
            )
            return scatter

        hemi_keys = sorted(list(surfs.keys()))

        for k in hemi_keys:
            assert k in ["lh", "rh"]

        cmap_sulci = plt.colormaps[sulci_cmap_kwargs["colormap"]]
        vmin_sulci = sulci_cmap_kwargs["vmin"]
        vmax_sulci = sulci_cmap_kwargs["vmax"]
        norm_sulci = Normalize(vmin=vmin_sulci, vmax=vmax_sulci)
        if sulci_cmap_nonlinearity is not None:
            cmap_sulci_func = lambda x: cmap_sulci(
                norm_sulci(sulci_cmap_nonlinearity(x))
            )

        else:
            cmap_sulci_func = lambda x: cmap_sulci(norm_sulci(x))

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
                if vmin is None:
                    if vmin is None:
                        vmin = elec_values.min()

                if vmax is None:
                    if vmax is None:
                        vmax = elec_values.max()

                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap_func = lambda x: cmap_overlay(norm(x))
                cmap_mappable = ScalarMappable(cmap=cmap_overlay, norm=norm)

        num_subfigs = len(hemi_keys)

        # do plotting on each axis if mpl, otherwise together
        axes = []
        for i, hemi in enumerate(hemi_keys):
            verts = surfs[hemi][0]
            triangles = surfs[hemi][1]
            sulc = sulci[hemi]

            if isinstance(view, str):
                if hemi == "lh":
                    elev, azim = self.lh.view(view, backend=backend)
                else:
                    elev, azim = self.rh.view(view, backend=backend)
            else:
                assert isinstance(view, tuple)
                elev, azim = view

            # color by sulci
            triangle_values_sulci = np.array(
                [[sulc[nn] for nn in triangles[i]] for i in range(len(triangles))]
            ).mean(1)
            colors_sulci = cmap_sulci_func(triangle_values_sulci)

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
                # set the face colors of the Poly3DCollection
                colors_sulci[:, -1] = brain_alpha
                p3dc.set_fc(colors_sulci)

            if elecs is not None:
                # snap elecs to pial surface
                if snap_to_surface:
                    if hemi == "lh":
                        map_surf = self.lh.surf_pial
                    else:
                        map_surf = self.rh.surf_pial
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

                if elec_values is None:
                    # if no values to map, use given colors
                    if colors is None:
                        elec_colors = np.zeros((len(coords), 4))
                        elec_colors[:, -1] = elec_alpha
                    elif isinstance(colors, str):
                        if isinstance(elec_alpha, (float, int)):
                            elec_colors = np.asarray(
                                [mpl.colors.to_rgba(colors, elec_alpha)]
                                * len(elec_isleft)
                            )
                        else:
                            elec_colors = np.asarray(
                                [
                                    mpl.colors.to_rgba(colors, alph)
                                    for alph in elec_alpha
                                ]
                            )
                    elif isinstance(colors, list):
                        if isinstance(elec_alpha, (float, int)):
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
                        if elec_colors.shape[1] > 3:
                            elec_colors[:, 3] = elec_alpha
                    else:
                        raise TypeError(
                            "no values given, and colors could not be interpreted as either numpy array, single color string, or list of strings"
                        )

                else:  # we do have values that we can map
                    if hemi == "lh":
                        elec_colors = cmap_func(elec_values)
                    else:
                        elec_colors = cmap_func(elec_values)

                # restrict to only this hemisphere
                if hemi == "lh":
                    coords = coords[elec_isleft]
                    elec_colors = elec_colors[elec_isleft]
                else:
                    coords = coords[~elec_isleft]
                    elec_colors = elec_colors[~elec_isleft]

                if backend == "plotly":
                    elec_colors *= 255
                    elec_colors = elec_colors.astype("int")
                    scatter = _plotly_scatter3d(
                        coords, elec_colors, name=f"elecs-{hemi}"
                    )
                    trace_list.append(scatter)
                else:  # mpl
                    x, y, z = coords.T
                    ax.scatter(x, y, z, s=elec_size, c=elec_colors, **kwargs)

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


def left2hemi(is_left: bool):
    """
    Converts boolean indicator of hemisphere to string tag 'lh' or 'rh'.

    Returns:
        hemi: hemisphere code string ('lh' or 'rh')
    """
    return "lh" if is_left else "rh"


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


def _tri_indices(simplices):
    # simplices is a numpy array defining the simplices of the triangularization
    # returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))
