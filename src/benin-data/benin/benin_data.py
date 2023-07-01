
# This file includes work covered by the following copyright and permission notices:
#
#  Copyright 2023 Google LLC
#  Licensed under the Apache License, Version 2.0 (the "License");


"""
Data retrieval code for Benin for SPIRES Lab Projects.


Modify this file to change what create_dataset.py 
creates the testing/training datset from.

Project patterns based on weather-forcasting sample:
https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting


"""

# For posponed evaluations
#from __future__ import annotations

import io
import ee
from google.api_core import exceptions, retry
import google.auth
import requests

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured


# Default globals
SCALE = 150


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
    opticalBands = ['B3','B2','B1'] #RGB
    thermalBands = ['B4','B3'] #NIR

    # Specify inputs (Landsat bands) to the model and the response variable.

    BANDS = ['R', 'G', 'B', 'NDVI']
    RESPONSE = 'target'

    benin = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na','Benin')).set('ORIG_FID',0)

    # Prepare the cloud masked LANDSAT 7 median composite image
    # clip it to the outline of Benin and then select the R,G,B, and NDVI bands.
    image = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR").filterDate('2007-01-01', '2008-12-31')
    image = (image.map(mask_clouds_landsat).median().clip(benin.geometry().buffer(10000)))
    image_ndvi = image.normalizedDifference(thermalBands).rename(['NDVI'])
    image_rgb = image.select(opticalBands).rename(['R','G','B'])
    image = image_rgb.addBands(image_ndvi)

    return image.unmask(0)


def get_labels_image(as_double:bool = True) -> ee.Image:
    """ Retrieves the labels image for Benin data

        - Creates a mask where areas in villages are 1
        and outisde are 0
        - Geometry created based on get_inputs_image
        return


    Args:
      as_double (bool): Whether to cast the image 
      to a double. This is done to match the 
      dtype of the input image

    Returns:
      labels image as ee.Image with 'target' band
    """

    #Import Voronoi Raster, get treated
    treated_voronoi = ee.FeatureCollection('projects/ls-test-3-24/assets/voronoi_villages')\
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
        scale=SCALE,
        region=benin.geometry(),
        seed=seed,
        geometries=True,
    )


    for point in points.toList(points.size()).getInfo():
        yield point["geometry"]["coordinates"]


def get_inputs_patch(point: tuple, patch_size: int) -> np.ndarray:
    """Gets the patch of pixels for the inputs.

    Args:
        point: A (longitude, latitude) coordinate.
        patch_size: Size in pixels of the surrounding square patch.

    Returns: The pixel values of a patch as a NumPy array.
    """
    image = get_inputs_image()
    patch = get_patch(image, point, patch_size, 30)
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
    patch = get_patch(image, point, patch_size, 30)
    return structured_to_unstructured(patch)

@retry.Retry(deadline=10 * 60)  # seconds
def get_patch(image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int) -> np.ndarray:
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



