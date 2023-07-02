# dd_changenotes:
GC == Google Cloud

Adjusted for [weather AI notebook](https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting)

## 6/29
Current issue in running the pipeline locally ("EEException: Invalid number of coordinates: 1 [while running '[165]: 📑 Get example']"). Currently stack tracing to try to discover issue.
- Added `SCALE`
- Added functionality to make `get_labels_image` return label as double to match float64 of input image (resolved [#5](#5))

## 6/30:
- Fixed visualization to reflect `get_*_image` functionality
- Removed weather-forcasting version of pipeline, replaced with original working pipeline (NumPy -> tfrecord)

## 7/1:
- Changed setup to clone SPIRES repo
- Refactored to use repo ^ packages (Added src/)
- Tested DataFlow with [create_dataset.py](create_dataset.py) script
 - Fixed [#4](#4)
- TODO: General refactoring, probably need stricter version requirement checks
- **Current issue**: `serialize_tensorflow` seems to be failing in DataFlow pipeline. See [failed jobs](https://console.cloud.google.com/dataflow/jobs?project=ls-test-3-24&authuser=0) in GC console.
 - Possible solution: packaging things better
 - TODO: test pipeline output locally, though no serialization errors when running local pipeline.  
- Another issue, though less important: Max workers for pipeline is 1 even though it's explicitly set as higher
 - Possible solution: idk; maybe GC setting