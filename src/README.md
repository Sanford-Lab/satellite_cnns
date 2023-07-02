# SPIRES Packages

## Organization
/src/                               Package directory

    ├── benin-data/
    |   ├── pyproject.toml
    │   └── benin                   Package for Benin project
    │     ├── __init__.py  
    │     └── bein_data.py          Functionality to pull data for Benin project
    ├── kenya-data/                 
    │   ├── pyproject.toml
    │   └── kenya                   Unfinished package for Kenya project
    │      ├── __init__.py          
    │      └── kenya_data.py        
    └── README.md                   Package info


## Info
Data packages are meant to hold the code to pull the appropriate data for a given project. 
The main functionality they need to achieve is `get_inputs_image->ee.Image`, 
`get_labels_image->ee.Image`, and `sample_points` (generator for coordinate tuples).
