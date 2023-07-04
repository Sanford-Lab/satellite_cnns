# torch.utils.data.Dataset

import os
import glob
from typing import Any
import numpy as np
from torch.utils.data import Dataset, random_split
from torch import Generator

# Constants.
NUM_DATASET_READ_PROC = 16  # number of processes to read data files in parallel
NUM_DATASET_PROC = os.cpu_count() or 8  # number of processes for CPU transformations


## NOT WORKING CURRENTLY
class DatsetFromPath(Dataset):
    """torch.utils.data.Dataset subclass for Benin Data"""
    
    def __init__(self, data_path : str) -> None:
        files = glob(os.path.join(data_path, "*.npz"))
        
        data : dict= {"filename": files}.map(
            self.read_data_file, 
            num_proc=NUM_DATASET_READ_PROC,
            remove_columns=["filename"]
        ).map(self.flatten, batched=True, num_proc=NUM_DATASET_PROC)
        
        self._inputs = data["inputs"]
        self._labels = data["labels"]
    
        
    def __getitem__(self, index) -> Any:
        return (self._inputs[index], self._labels[index])
    
    def __len__(self) -> int:
        return len(self._inputs) 
    
    def read_data_file(item: dict[str,str]) -> dict[str, np.ndarray]:
        with open(item["filename"], "rb") as f:
            npz = np.load(f)
            return {"inputs": npz["inputs"], "labels": npz["labels"]}
    
    def flatten(batch: dict) -> dict:
        return {key: np.concatenate(values) for key, values in batch.items()}

def read_dataset(data_path: str, train_test_ratio: float, seed=0) -> tuple:
    assert(train_test_ratio <= 1)
    dataset = DatsetFromPath(data_path)
    
    seed_gen = Generator().manual_seed(seed)
    return random_split(
        dataset=dataset,
        lengths=[train_test_ratio, (1-train_test_ratio)],
        generator=seed_gen)
    