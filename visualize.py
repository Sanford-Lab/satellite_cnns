# This file includes work covered by the following copyright and permission notices:
#
#  Copyright 2023 Google LLC
#  Licensed under the Apache License, Version 2.0 (the "License");
#
# =============================================================================================================
#
# Module to visualize data and patches for for SPIRES Lab Projects.
#
# 
#
# Project patterns based on weather-forcasting sample:
# https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting
#
#______________________________________________________________________________________________________________


import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from benin.data import get_inputs_image, get_labels_image

def show_patches(inputs_patch: np.ndarray, labels_patch: np.ndarray)-> None: 
    """Shows patch plot. Blue = inside village, red = outside village 

    Args:
        inputs_patch (np.ndarray): get this from get_input_patch
        labels_patch (np.ndarray): get this from get_labels_patch
    """
    inputs_visual = inputs_patch[:,:, :3]
    inputs_visual = inputs_visual/np.amax(inputs_visual)
    fig, axs = plt.subplots(1, 2, layout='constrained')

    cmap = colors.ListedColormap(['blue', 'red'])
    bounds=[0,1,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    #plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 5, 10])
    axs[0].imshow(inputs_visual)
    axs[1].imshow(labels_patch, origin='lower', interpolation='nearest', cmap=cmap, norm=norm)
    axs[0].set_title("Input image")
    axs[1].set_title("Labels")
    plt.show()
    

# maybe add visualization for sampled patches