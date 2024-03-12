"""
============================
Plot Intracranial Electrodes
============================

This notebook demonstrates calculating anatomical information about
intracranial electrodes and plotting on the Freesurfer average brain.

"""
# Author: Gavin Mischler
# 
# License: MIT

import numpy as np
import matplotlib.pyplot as plt

import naplib as nl

###############################################################################
# Load freesurfer fsaverage data if we don't have it

import os
import mne
os.makedirs('./fsaverage', exist_ok=True)
mne.datasets.fetch_fsaverage('./fsaverage/')

###############################################################################
# Create a brain with the pial surface for computing metrics
brain = nl.Brain('pial', subject_dir='./fsaverage/').split_hg('midpoint').split_stg().simplify_labels()

# Specify the coordinates of 30 electrodes in fsaverage space
coords = np.array([[-47.281147  ,  17.026093  , -21.833099  ],
                   [-48.273964  ,  16.155487  , -20.162935  ],
                   [-51.101261  ,  13.711058  , -16.258459  ],
                   [-55.660889  ,   9.761111  , -12.340655  ],
                   [-58.733326  ,   6.046287  ,  -9.626602  ],
                   [-60.749279  ,   2.233287  ,  -8.044459  ],
                   [-61.26712   ,  -1.939675  ,  -8.582445  ],
                   [-63.686226  , -10.447982  ,  -0.445693  ],
                   [-63.453224  ,  -9.826311  ,   1.095302  ],
                   [-48.792809  ,  15.73144   , -19.34193   ],
                   [-51.336754  ,  13.27527   , -15.57861   ],
                   [-53.301971  ,  11.016301  , -12.48259   ],
                   [-55.044659  ,   9.894337  , -11.228349  ],
                   [-57.597462  ,   6.753941  ,  -8.082416  ],
                   [-60.594891  ,   2.579503  ,  -6.884331  ],
                   [-63.078999  ,  -8.770401  ,  -1.878142  ],
                   [-67.419235  , -26.153931  ,  -1.260003  ],
                   [-60.28742599, -11.71243477,   5.62593937],
                   [-63.12403107, -12.37896156,   4.09772062],
                   [ 64.44213867,  -3.16063929,  -6.95104313],
                   [ 61.58537674, -23.53317833,  -3.20349312],
                   [ 69.31034851, -18.18317795,   1.97798777],
                   [ 69.0439682 , -18.64465904,   1.2625511 ],
                   [ 68.32962799, -20.90372849,  -0.25190961],
                   [ 59.79437256, -23.76178932,  -3.52095652],
                   [-56.57900238,  -9.23060513,  -7.33194447],
                   [-58.861763  , -11.2859602 ,  -6.18047237],
                   [-61.13874054, -11.35863781,  -4.49999475],
                   [-60.82435989,  -8.91696072,  -3.20156574],
                   [-61.00576019,  -7.45676041,  -3.06485367]])

isleft = coords[:,0]<0

###############################################################################
# Get anatomical labels for each electrode

anatomical_labels = brain.annotate_coords(coords, isleft)
print(anatomical_labels)

###############################################################################
# Compute the distance of each electrode from posteromedial HG along the
# cortical surface

dist_from_HG = brain.distance_from_region(coords, isleft, region='pmHG', metric='surf')
print(dist_from_HG)

###############################################################################
# Create a brain with the inflated surface for plotting
brain = nl.Brain('inflated', subject_dir='./fsaverage/').split_hg('midpoint').split_stg().simplify_labels()

###############################################################################
# Plot electrode locations with matplotlib, and color the electrodes
# by their distance from HG

fig, axes = brain.plot_brain_elecs(coords, isleft, values=dist_from_HG, hemi='lh', view='lateral')
plt.show()

###############################################################################
# Plot electrodes with an interactive plotly figure, and instead of coloring them by a value, we
# will color them by a custom color for each electrode.
# Some common use-cases for this might be to color electrodes based on the identity of the subject
# they came from when pooling electrodes, or by a categorical variable. In this case, we color
# them by the anatomical labels they were assigned to (black for posterior STG and red for middle STG)

colors = ['k' if lab == 'pSTG' else 'r' for lab in anatomical_labels]
fig, axes = brain.plot_brain_elecs(coords, isleft, colors=colors, backend='plotly')
fig.write_html("interactive_brain_plot.html") # save as an interactive html plot
fig.show() # show the interactive plot in the notebook

###############################################################################
# Color certain brain regions by their label
brain.paint_overlay('mSTG', -3)
brain.paint_overlay('pSTG', 3)
brain.paint_overlay('MTG', 1)
fig, axes = brain.plot_brain_overlay(view='lateral')
plt.show()


