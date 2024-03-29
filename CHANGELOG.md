# dd_changenotes:
GC == Google Cloud
GS == Google Storage

Adjusted for [weather AI notebook](https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/people-and-planet-ai/weather-forecasting)

## Currently Working on: custom PyTorch Datset class and Subset functionality

## To Do
[ ] Now that we're using NPZ files, test if we need to convert labels to float64 (`as_double` functionality in Benin [data.py](src/benin-data/benin/data.py))\
[ ] Change all float64s to float32s to match better with Pytorch (create_dataset change please)\
[ ] Adjust train.py to support different dataset shapes (num of input and label bands)\
[ ] Add logger in train.py\
[x] Read/test uploaded NPZ files\


## Future improvements
[ ] Support for TensorFlow framework\
[ ] Parallel processing for reading data files (read_data.py)\
[ ] Weather sample augments it's examples, good to implement in future training
[ ] Save created model directly to VertexAI
  - Ref "Training did not produce a Managed Model returning None. Training Pipeline projects/978289642310/locations/us-central1/trainingPipelines/5677616430787330048 is not configured to upload a Model. Create the Training Pipeline with model_serving_container_image_uri and model_display_name passed in. Ensure that your training script saves to model to os.environ['AIP_MODEL_DIR']."


## 7/6:
- Continued work on U-Net implementation and train.py
- Produced a PyTorch model with train.py locally (model training effectiveness untested)
  - finished proof of concept implementation of benin-model package
- Ran traning on VertexAI
  - took npz files from GS bucket 
- Added more helpful to model
- Added basic notebook annotaiton update for training demonstration

## 7/5:
- Started Pytorch U-Net implementation and package (benin-model) to use for training (train.py)
  - Moving away from weather forcasting implementation (more object oriented) to avoid using Hugging Face Trainer. Instead implementing U-Net (https://github.com/milesial/Pytorch-UNet)

## 7/4:
- Finished and tested torch.utils.data.Dataset custom subclass (DatasetFromPath)
- Tested workflow up to use of DatasetFromPath and splitting the dataset
 - Modified notebook to reflect work

## 7/3:
- Created library [read_data.py](read_data.py) for reading the data in NPZ format
- Began translation of weather forcasting reading NPZ, read_dataset to avoid use of Hugging Face Datasets

## 7/2:
- Updated package package file names (benin_data.py -> data.py, etc)
- Changed `SCALE` in [benin/data.py](src/benin-data/benin/data.py) to split to `SAMPLE_SCALE` and `PATCH_SCALE`
  - Stratified sampling at too low of a scale is very memory intenisve and that kind of precision isn't needed. We can increase the strat sampling scale as long as we make sure that the patches retireved from EE is scaled... to the scale (ex: if patch scale is 10, sample scale can be 10,100,1000, etc)
- Cleaned up [create_dataset.py](create_dataset.py)
  - Specified what should be customized to change defaults
- Tested inputs and labels patch using new `show_patches` function (to be added to a visualize.py file)
- Added and tested new pipeline in [create_dataset.py](create_dataset.py)
  - Uses compressed NumPy files instead of TFrecord. This fixes the missing tensorflow module using Dataflow (#4) because it isn't required.
  - WORKING DISTRIBUTED RUNNER
- Changed [benin_apache_pipeline.ipynb](benin_apache_pipeline.ipynb) to reflect current project state.

## 7/1:
- Changed setup to clone SPIRES repo
- Refactored to use repo ^ packages (Added src/)
- Tested DataFlow with [create_dataset.py](create_dataset.py) script
  - Fixed [#4](/../../issues/4) with built and attached data package
- TODO: General refactoring, probably need stricter version requirement checks
- **Current issue**: `serialize_tensorflow` seems to be failing in DataFlow pipeline. See [failed jobs](https://console.cloud.google.com/dataflow/jobs?project=ls-test-3-24&authuser=0) in GC console.
  - Possible solution: packaging things better
  - TODO: test pipeline output locally, though no serialization errors when running local pipeline.  
- Another issue, though less important: Max workers for pipeline is 1 even though it's explicitly set as higher
  - Possible solution: idk; maybe GC setting

## 6/30:
- Fixed visualization to reflect `get_*_image` functionality
- Removed weather-forcasting version of pipeline, replaced with original working pipeline (NumPy -> tfrecord)


## 6/29
Current issue in running the pipeline locally ("EEException: Invalid number of coordinates: 1 [while running '[165]: 📑 Get example']"). Currently stack tracing to try to discover issue.
- Added `SCALE`
- Added functionality to make `get_labels_image` return label as double to match float64 of input image (resolved [#5](#5))
