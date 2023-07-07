# SPIRES Packages

## Organization
/src/                               Package directory

    ├── benin-data/
    |    ├── pyproject.toml
    │    └── benin                   Package for Benin project
    │        ├── data.py             Functionality to pull data for Benin project
    │        └── __init__.py         
    ├── benin-model/                 
    │    ├── pyproject.toml
    │    └── benin                   Modeling package for Benin project
    │        ├── dice_score.py       Holds dice score helper functions for training
    │        ├── model.py            Defined U-Net model and layers
    │        ├── npz_dataset.py      Module library for instantiating a torch Dataset from .npz files in a directory
    │        ├── train.py            Package main: script that uses other modules to create, train, and save a model based on args/defaults
    │        └── __init__.py
    ├── kenya-data/                 
    │    ├── pyproject.toml
    │    └── kenya                   Unfinished package for Kenya project
    │        ├── data.py          
    │        └── __init__.py
    └── README.md                    Package info


## Info
Data packages are meant to hold the code to pull the appropriate data for a given project. 
The main functionality they need to achieve is `get_inputs_image->ee.Image`, 
`get_labels_image->ee.Image`, and `sample_points` (generator for coordinate tuples).

Model packages aren't yet refactored for universality.
