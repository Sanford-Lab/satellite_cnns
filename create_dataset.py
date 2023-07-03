# This file includes work covered by the following copyright and permission notices:
#
#  Copyright 2023 Google LLC
#  Licensed under the Apache License, Version 2.0 (the "License");
#
# =============================================================================================================
#
# Script to create_dataset for SPIRES Lab Projects.
#
# This script is designed to adapt to different countries
#
# Project patterns based on weather-forcasting sample:
# https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting
#
#______________________________________________________________________________________________________________

# For posponed evaluations
from __future__ import annotations


# ========================================= #
#   CHANGE THIS CODE DEPENDING ON PROJECT   #
# +++++++++++++++++++++++++++++++++++++++++ #
#                                           #
#   Change import depending on data using:  #
from benin.data import sample_points, \
    get_inputs_patch, get_labels_patch      #
#                                           #
#       Change default globals:             #
POINTS_PER_CLASS = 4                        #
PATCH_SIZE = 128                            #
MAX_REQUESTS = 20                           #
MIN_BATCH_SIZE = 100                        #
#                                           #
# ========================================= #


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #
#              The rest of the code shouldn't need to be altered            #
#                                                                           #
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions

import logging
import numpy as np
import uuid


def get_training_example(lonlat, patch_size = 128) -> tuple:
    # Ref #14: Change when implemented
    #from benin.benin.data import get_inputs_patch, get_labels_patch
    
    return (
    	get_inputs_patch(lonlat, patch_size), get_labels_patch(lonlat, patch_size)
    )

def write_npz(batch: list[tuple[np.ndarray, np.ndarray]], data_path: str) -> str:
    """Writes an (inputs, labels) batch into a compressed NumPy file.
        - taken from weather-forcasting sample
    Args:
        batch: Batch of (inputs, labels) pairs of NumPy arrays.
        data_path: Directory path to save files to.
    Returns: The filename of the data file.
    """
    filename = FileSystems.join(data_path, f"{uuid.uuid4()}.npz")
    with FileSystems.create(filename) as f:
        inputs = [x for (x, _) in batch]
        labels = [y for (_, y) in batch]
        np.savez_compressed(f, inputs=inputs, labels=labels)
    return filename

def run(
    data_path: str,
    points_per_class: int = POINTS_PER_CLASS,
    patch_size: int = PATCH_SIZE,
    max_requests: int = MAX_REQUESTS,
    beam_args: Optional[List[str]] = None,
) -> None:
    
    beam_options = PipelineOptions(
        beam_args,
        save_main_session=True,
        max_num_workers=max_requests,  # distributed runners
        direct_num_workers=max(max_requests, MAX_REQUESTS),  # direct runner
    )
    with beam.Pipeline(options=beam_options) as pipeline:
        (
            pipeline
            | "🌱 Make seeds" >> beam.Create([0])
            | "📌 Sample points" >> beam.FlatMap(sample_points, points_per_class=points_per_class)
            | "🃏 Reshuffle" >> beam.Reshuffle()
            | "🛰 Get examples" >> beam.Map(get_training_example, patch_size=patch_size)
            | "🗂️ Batch examples" >> beam.BatchElements()
            | "📝 Write NPZ files" >> beam.Map(write_npz, data_path)
        )


def main() -> None:
    """Main script for creating dataset in DataFlow

    Parses through arguments given to script to upload compressed NumPy files

    """
    
    import argparse
    
    logging.getLogger().setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data-path",
        required=True,
        help="Directory path to save the data files",
    )
    parser.add_argument(
        "--ppc",
        type=int,
        default=POINTS_PER_CLASS,
        help="Points per class for strat sampling.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=MAX_REQUESTS,
        help="Limit the number of concurrent requests to Earth Engine.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"Patch size for image. Default is {PATCH_SIZE}.",
    )
    args, beam_args = parser.parse_known_args()

    run(
        data_path=args.data_path,
        points_per_class=args.ppc,
        max_requests=args.max_requests,
        patch_size=args.patch_size,
        beam_args=beam_args,
    )



if __name__ == "__main__":
    main()

