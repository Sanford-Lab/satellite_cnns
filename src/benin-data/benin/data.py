# This file includes work and structure covered by the following copyright and permission notices:
#
#  Copyright 2023 Google LLC
#  Licensed under the Apache License, Version 2.0 (the "License");
#
# ==============================================================================================================
#
# Data retrieval code for Benin for SPIRES Lab Projects.
#
#
# Modify this file to change what create_dataset.py creates the testing/training datset from.
#
# Project patterns based on weather-forcasting sample:
# https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting
#
# ______________________________________________________________________________________________________________



# For posponed evaluations
#from __future__ import annotations

import io
import ee
from google.api_core import exceptions, retry
import google.auth
import requests
from typing import Iterable

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured


# Authenticate and initialize Earth Engine with the default credentials.
credentials, project = google.auth.default(
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/earthengine",
    ]
)

# Use the Earth Engine High Volume endpoint.
#   https://developers.google.com/earth-engine/cloud/highvolume
ee.Initialize(
    credentials.with_quota_project(None),
    project=project,
    opt_url="https://earthengine-highvolume.googleapis.com",
)


# Default globals
PATCH_SCALE = 10 
SAMPLE_SCALE = 1000
assert(SAMPLE_SCALE % PATCH_SCALE == 0) # PATCH_SCALE should be multiple of SAMPLE_SCALE

#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# The following code should be edited depending on the project.
# 
# To be a functional file for creating a dataset, define
#   - get_inputs_image()-> ee.Image
#   - get_labels_image()-> ee.Image
#   - sample_points(seed: int, points_per_class: int) -> Iterable[tuple[float, float]]
#
#OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

def mask_clouds_landsat(image: ee.Image) -> ee.Image:
    """
    Function credit unknown @LS

    """
    # Bits 3 and 5 are cloud shadow and cloud, respectively
    cloudShadowBitMask = (1 << 3) # 1000 in base 2
    cloudsBitMask = (1 << 5) # 100000 in base 2

    # Get the pixel QA band
    qa = image.select('pixel_qa')

    #Both flags should be set to zero, indicating clear conditions
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).bitwiseAnd(qa.bitwiseAnd(cloudsBitMask).eq(0))

    # Mask image with clouds and shadows
    return image.updateMask(mask)

def get_inputs_image() -> ee.Image:
    """ Retrieves the input image for Benin data

    Uses Landsat7

    Args:
      as_double (bool): Whether to cast the image 
          to a double. This is done to match the 
          dtype of the input image

    Returns:
      ee.Image: inputs with bands 'R', 'G', 'B', 'NDVI'
        - has benin geometry with 10000 buffer

    """

    def prep_sr_l7(image):
        # Develop masks for unwanted pixels (fill, cloud, cloud shadow).
        qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
        
        # Apply the scaling factors to the appropriate bands.
        def get_factor_img(factor_names):
            factor_list = image.toDictionary().select(factor_names).values()
            return ee.Image.constant(factor_list)
        
        scale_img = get_factor_img(['REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B6'])
        offset_img = get_factor_img(['REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B6'])
        scaled = image.select('SR_B.|ST_B6').multiply(scale_img).add(offset_img)
        
        # Replace original bands with scaled bands and apply masks.
        return image.addBands(scaled, None, True).updateMask(qa_mask)

    # Define the date range for the filter
    start_date = '2006-01-01'
    end_date = '2008-12-31'
    
    # Specify inputs (Landsat bands) to the model and the response variable
    optical_bands = ['SR_B3', 'SR_B2', 'SR_B1']  # RGB
    nir_band = 'SR_B4'  # NIR
    
    # Grab the Benin feature (shape of country)
    benin_shape = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', 'Benin')).first()
    
    # Filter and process the Landsat 7 Collection 2 dataset
    l7_filtered = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
        .filterDate(start_date, end_date) \
        .map(prep_sr_l7) \
        .median() \
        .clip(benin_shape.geometry().buffer(10000))
    
    # Create NDVI band, rename RGB, and combine
    benin_input = l7_filtered.select(optical_bands + [nir_band]) \
        .rename(['R', 'G', 'B', 'NIR']) \
        .addBands(l7_filtered.normalizedDifference([nir_band, 'SR_B3']).rename('NDVI'))

    return benin_input.unmask(0) 


def get_labels_image(as_double:bool = True) -> ee.Image:
    """ Retrieves the labels image for Benin data
    
        - Uses voronoi asset 'projects/satellite-cnns/assets/benin/voronoi_villages'
        - Creates a mask where areas in villages are 1 and outside are 0
        - Geometry created based on get_inputs_image return


    Args:
      as_double (bool): Whether to cast the image 
      to a double. This is done to match the 
      dtype of the input image

    Returns:
      labels image as ee.Image with 'target' band
    """

    # Import vornoi a ('satellite-cnns/assets/voronoi_villages')
    # and convert to feature collection
    treated_voronoi = ee.FeatureCollection('projects/satellite-cnns/assets/benin/voronoi_villages')\
                                .filter(ee.Filter.eq('treated', 1))

    # Create a village mask based on the treated village raster
    villagemask = treated_voronoi.filter(ee.Filter.notNull(['VID']))\
                                .reduceToImage(properties=['VID'],reducer= ee.Reducer\
                                .first())\
                                .mask()

    # Create the target image using the village mask prepared earlier.
    inputs = get_inputs_image()
    l7Masked = inputs.updateMask(villagemask)

    # This part is hacky. How could we do it better?
    l7Unmasked = l7Masked.unmask(-9999)
    outside_circle = 'b("R") > -9000'
    target = l7Unmasked.expression(outside_circle).rename("target")

    if as_double: target = target.double()

    return target


def sample_points(seed: int = 0, points_per_class = 2) -> Iterable[tuple[float, float]]:
    """ Generates coordinate tuple of coordinates for sampling based on the labels image
        - Order of point generatiom: inside outside villages then in

    Args:
      seed (int): seed for stratifiedSample
      points_per_class (int): number of points per class in get_labels_image return

    Returns:
        Itterable of coordinates (tuple[float,float])
    """

    target = get_labels_image(as_double=False)
    benin = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na','Benin')).set('ORIG_FID',0)

    points = target.stratifiedSample(
        points_per_class,
        scale=SAMPLE_SCALE,
        region=benin.geometry(),
        seed=seed,
        geometries=True,
    )

    for point in points.toList(points.size()).getInfo():
        yield point["geometry"]["coordinates"]





#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# The code below should not need to change between projects (how patches are retrieved)            #
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


def get_inputs_patch(point: tuple, patch_size: int) -> np.ndarray:
    """Gets the patch of pixels for the inputs.

    Args:
        point: A (longitude, latitude) coordinate.
        patch_size: Size in pixels of the surrounding square patch.

    Returns: The pixel values of a patch as a NumPy array.
    """
    image = get_inputs_image()
    patch = get_patch(image, point, patch_size, PATCH_SCALE)


    return structured_to_unstructured(patch)

def get_labels_patch(point: tuple, patch_size: int) -> np.ndarray:
    """Gets the patch of pixels for the labels.

    Args:
        point: A (longitude, latitude) coordinate.
        patch_size: Size in pixels of the surrounding square patch.

    Returns:
        The pixel values of a patch as a NumPy array.
    """
    image = get_labels_image()
    patch = get_patch(image, point, patch_size, PATCH_SCALE)


    return structured_to_unstructured(patch)


@retry.Retry(deadline=10 * 60)  # Requires retries to work within 10 min
def get_patch(image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int) -> np.ndarray:
    """Gets the patch of pixels based on parameters.

    Args:
        point: A (longitude, latitude) coordinate.
        patch_size: Size in pixels of the surrounding square patch.

    Returns:
        The pixel values of a patch as a NumPy array.
    """
    geometry = ee.Geometry.Point(lonlat)
    url = image.getDownloadURL({
        "region": geometry.buffer(scale * patch_size / 2, 1).bounds(1),
        "dimensions": [patch_size, patch_size],
        "format": "NPY",
    })

    # If we get "429: Too Many Requests" errors, it's safe to retry the request.
    # The Retry library only works with `google.api_core` exceptions.
    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)

    # Still raise any other exceptions to make sure we got valid data.
    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True)



