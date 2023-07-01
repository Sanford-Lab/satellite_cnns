
# This file includes work covered by the following copyright and permission notices:
#
#  Copyright 2023 Google LLC
#  Licensed under the Apache License, Version 2.0 (the "License");


"""
Data retrieval code for Benin for SPIRES Lab Projects


Modify this file to change what create_dataset.py 
creates the testing/training datset from.

Project patterns based on weather-forcasting sample:
https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting


"""

# For posponed evaluations
#from __future__ import annotations

import ee
from google.api_core import exceptions, retry
import google.auth
import requests



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


