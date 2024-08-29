#
#   This module allows you to create a PyTorch torch.utils.data.Dataset from 
#   a path that contains NumPy compressed files (NPZ) files
#
#       -   Expects *.npz.files to be 'inputs' and 'labels'
#       -   train_test_split: splits DatasetFromPath into a custom subset 
#           (CustomSubset) for more convenient data access
#       -   Currently should be good to go for different projects as long as
#           npz files are in correct format
#
#   SPIRES Lab, 2023
#
#   Note: This is based on a copy of W/ npz_dataset.py and is frequently updated.
# ______________________________________________________________________________________________________________


import os
from glob import glob
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Union,
    TypeVar,
    Callable
)

import numpy as np
from torch.utils.data import Dataset
from torch import Generator, randperm, Tensor
from itertools import accumulate

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

# Default values
SEED = 0
TRAIN_TEST_RATIO = 0.8

class DatasetFromPath(Dataset):
    """torch.utils.data.Dataset custom subclass
    Expects: data_path to directory containing .npz files where the NumPy
    files are named 'inputs' and 'labels' containing respective np.ndarrays 
    of shape (# in batch, PATCH_SIZE, PATCH_SIZE, # of bands)
    """
    
    def __init__(self, data_path : str) -> None:
        files = glob(os.path.join(data_path, "*.npz"))
        if len(files) <= 0:
            raise EmptyDirectoryError(data_path)
    
        first_file = np.load(files[0])
        inputs, labels = first_file['inputs'], first_file['labels']
        files = files[1:]
        for f in files:
            loaded = np.load(f)
            inputs, labels = np.vstack((inputs, loaded['inputs'])), np.vstack((labels, loaded['labels']))

        # We're going to put it to float32 to be consistant w NumPy floats
        self._inputs = np.float32(inputs)
        self._labels = np.float32(labels)
        self._ogpath = data_path
        
        
    def get_inputs(self) -> np.ndarray:
        return self._inputs
    
    def get_labels(self) -> np.ndarray:
        return self._labelsv
        
    def __getitem__(self, index) -> Any:
        if(isinstance(index, int)):
        # Using dictionary support like in weather forecasting sample
            return {'inputs': self._inputs[index], 'labels' : self._labels[index]}
        elif isinstance(index, str) & (index == 'inputs' or index == 'labels'):
            return self._inputs if index == 'inputs' else self._labels
        
    
    def __len__(self) -> int:
        return len(self._inputs) 
    
    def __str__(self) -> str:
        return f'Dataset from \'{self._ogpath}\', size: {self.__len__()}, inputs: {self._inputs.shape}, labels: {self._labels.shape}'
    

def train_test_split(data:DatasetFromPath, ratio:float = TRAIN_TEST_RATIO, seed = SEED, transform_dict:dict[str, Callable[..., Tensor]] | None = None) -> tuple:
    """Splits the given dataset by the ratio given. Implements a random split per 
    torch.utils.data.randomsplit

    Args:
        data (DatasetFromPath): dataset
        ratio (float, optional): ratio between 0 and 1 inclusive. Defaults to TEST_TRAIN_RATIO.
        seed (_type_, optional): seed used for shuffling a random split. Defaults to SEED.
        transform (dict[str, Callable[..., Tensor]] | None, optional): Dictionary of transform functions for train and test dataset. Default to None.

    Returns:
        tuple: (test subset, train sub) both as torch.utils.data.Subset objects
    """
    assert(ratio >=0 and ratio <= 1)
    seed_gen = Generator().manual_seed(SEED)

    if not transform_dict:
        return custom_random_split(dataset=data, lengths=[round(ratio,2), round(1-ratio,2)], generator=seed_gen)
    else:
        train, test = custom_random_split(dataset=data, lengths=[round(ratio,2), round(1-ratio,2)], generator=seed_gen)

        train = CustomTransformedDataset(train, transform_dict['train'])
        test = CustomTransformedDataset(test, transform_dict['test'])

        return train, test


class EmptyDirectoryError(Exception):
    
    def __init__(self, path,
                 message=f'Path is empty or does not contain files ending in .npz') -> None:
        self.message = message + f'\tPath given: \"{path}\"'
        super().__init__(self.message)
        

class CustomSubset(Dataset[T_co]):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        elif isinstance(idx, str) & (idx == 'inputs' or idx == 'labels'):
            return self.dataset[idx][[i for i in self.indices]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CustomTransformedDataset(CustomSubset):
    def __init__(self, dataset: CustomSubset, transform = None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if(isinstance(index, int)):
            input = self.dataset[index]['inputs']
            label = self.dataset[index]['labels']
        
            if self.transform:
                input, label = self.transform(input, label)
            
            return {'inputs': input, 'labels' : input}
        
        elif isinstance(index, str) & (index == 'inputs' or index == 'labels'):
            if self.transform:
                # The v2 transforms generally accept an arbitrary number of leading dimensions (..., C, H, W),
                # whereas the shape of dataset[index] is (# in batch, PATCH_SIZE, PATCH_SIZE, # of bands).
                return self.transform(np.transpose(self.dataset[index], (0, 3, 1, 2)))
            else:
                return self.dataset[index]
        

def custom_random_split(dataset: Dataset[T], lengths: Sequence[Union[int, float]],
                 generator: Optional[Generator] = Generator().manual_seed(0)) -> List[CustomSubset[T]]:
    """
    Same as torch.utils.data.dataset.random_split except returns custom subclass and 
    and uses Generator().manual_seed(0) as default generator
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if np.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                np.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                print(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [CustomSubset(dataset, indices[offset - length : offset]) for offset, length in zip(accumulate(lengths), lengths)]