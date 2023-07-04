# torch.utils.data.Dataset


from typing import Any

from torch.utils.data import Dataset, random_split
from torch import Generator

# Constants.
SEED = 0
TEST_TRAIN_RATIO = 0.66
"""
Planning Notes
- We want to take in a data_path and train/test ratio to return a
    dictionary, dataDict:
    - dataDict keys are "train" and "test" with their values as 
    torch.utils.data.Dataset(s) representing a split torch.utils.data.Dataset
    for the combined numpy arrays in data_path
- Input: data_path, Ouput: dataDict ^
1) Get file names from data_path using glob
2) Use file names to load the NPZ using np.load
    - this has two numpy files in it ('inputs', 'labels')
    - npz['inputs'] is a numpy array with the inputs (#, 128,128,4)
    - npz['labels'] is (#, 128,128,1)
        - where # is the number of patches
    * What we ultimately want to do is to vstack these input and
    labels arrays for each file
        - Greedy approach: create a list of arrays, use vstack (use generator?) 

"""

class DatasetFromPath(Dataset):
    """torch.utils.data.Dataset custom subclass
    Expects: data_path to directory containing .npz files where the NumPy
    files are named 'inputs' and 'labels' containing respective np.ndarrays 
    of shape (# in batch, PATCH_SIZE, PATCH_SIZE, # of bands)
    """
    
    def __init__(self, data_path : str) -> None:
        import os
        from glob import glob
        files = glob(os.path.join(data_path, "*.npz"))
        assert(len(files) > 0)
        first_file = np.load(files[0])
        inputs, labels = first_file['inputs'], first_file['labels']
        files = files[1:]
        for f in files:
            loaded = np.load(f)
            inputs, labels = np.vstack((inputs, loaded['inputs'])), np.vstack((labels, loaded['labels']))

        self._inputs = inputs
        self._labels = labels
    
        
    def __getitem__(self, index) -> Any:
        # Using dictionary approach like in weather forecasting sample
        return {'inputs': self._inputs[index], 'labels' : self._labels[index]}
    
    def __len__(self) -> int:
        return len(self._inputs) 
    
def test_train_split(set:DatasetFromPath, ratio:float = TEST_TRAIN_RATIO, seed = SEED)-> tuple:
    assert(ratio >=0 and ratio <= 1)
    seed_gen = Generator().manual_seed(SEED)
    
    return random_split(dataset=set, lengths=[ratio, (1-ratio)], generator=seed_gen)
