# This file includes work covered by the following copyright and permission notices:
#
#  Copyright 2023 Google LLC
#  Licensed under the Apache License, Version 2.0 (the "License");


"""
Script to create_dataset for SPIRES Lab Projects.

This script is designed to adapt to different countries

Project patterns based on weather-forcasting sample:
https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting


"""

# For posponed evaluations
from __future__ import annotations


import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions
import logging

import ee

# Change import depending on data using
from benin.benin_data import sample_points


# Default globals
POINTS_PER_CLASS = 3
PATCH_SIZE = 128
MAX_REQUESTS = 20

def serialize_tensorflow(inputs: np.ndarray, labels: np.ndarray) -> bytes:
    """Serializes inputs and labels NumPy arrays as a tf.Example.

    Both inputs and outputs are expected to be dense tensors, not dictionaries.
    We serialize both the inputs and labels to save their shapes.

    Args:
        inputs: The example inputs as dense tensors.
        labels: The example labels as dense tensors.

    Returns: The serialized tf.Example as bytes.
    """
    import tensorflow as tf

    features = {
        name: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy()])
        )
        for name, data in {"inputs": inputs, "labels": labels}.items()
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def get_training_example(lonlat, patch_size = 128):
	from benin import get_inputs_patch, get_labels_patch

    return (
        get_inputs_patch(lonlat, patch_size),
        get_labels_patch(lonlat, patch_size),
    )

def run_tensorflow(
    data_path: str,
    points_per_class: int = POINTS_PER_CLASS,
    patch_size: int = PATCH_SIZE,
    max_requests: int = MAX_REQUESTS,
    beam_args: Optional[List[str]] = None,
) -> None:

    """Runs an Apache Beam pipeline to create a dataset.

	Function taken from Google land-classification sample

    This fetches data from Earth Engine and creates a TFRecords dataset.
    We use `max_requests` to limit the number of concurrent requests to Earth Engine
    to avoid quota issues. You can request for an increas of quota if you need it.

    Args:
        data_path: Directory path to save the TFRecord files.
        points_per_class: Number of points to get for each classification.
        patch_size: Size in pixels of the surrounding square patch.
        max_requests: Limit the number of concurrent requests to Earth Engine.
        polygons: List of polygons to pick points from.
        beam_args: Apache Beam command line arguments to parse as pipeline options.
    """
    

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
          | "🛰 Sample points" >> beam.FlatMap(sample_points) # uses scale of 150
          | "📖 Get examples" >> beam.Map(get_training_example)
          | "✍🏽 Serialize" >> beam.MapTuple(serialize_tensorflow)
          | "📚 Write TFRecords" >> beam.io.WriteToTFRecord(
              f"{data_path}/part", file_name_suffix=".tfrecord.gz"
          )
      )



def main() -> None:
	""" 
	Main script for creating dataset in DataFlow

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
        "--min-batch-size",
        type=int,
        default=MIN_BATCH_SIZE,
        help="Minimum number of examples to write per data file.",
    )
    args, beam_args = parser.parse_known_args()

    run_tensorflow(
        data_path=args.data_path,
        num_dates=args.num_dates,
        num_bins=args.num_bins,
        max_requests=args.max_requests,
        min_batch_size=args.min_batch_size,
        beam_args=beam_args,
    )





if __name__ == "__main__":
    main()








