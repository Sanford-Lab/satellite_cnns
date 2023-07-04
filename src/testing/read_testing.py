
#
# Unfinished testing file for read_data.py
# 
# Might be helpful to finish later for debugging but
#
#

import os
from glob import glob
import numpy as np

from create_dataset import write_npz
from create_dataset import get_training_example


PATCH_SIZE = 128
points = [[2.474858607749282, 12.03293323078099],
            [2.241296633878206, 11.197500016549837],
            [1.899936825912788, 7.325761141994698]]
# ===============================================================
files = glob(os.path.join('/content/testing-data', "*.npz"))


def read_data_file(filename) -> dict[str, np.ndarray]:
  with open(filename, "rb") as f:
      npz = np.load(f)
      return {"inputs": npz["inputs"], "labels": npz["labels"]}

def flatten(batch: dict) -> dict:
  return {key: np.concatenate(values) for key, values in batch.items()}


test_file = write_npz([get_training_example(p) for p in points], "/src/testing/dump")
print("Batch:")
load2 = np.load(test_file)
print(f"Inputs shape (# in batch, dim1, dim2, dim3): {load2['inputs'].shape}")
print(f"Labels shape (# in batch, dim1, dim2, dim3): {load2['labels'].shape}")
print("Confirm that flattened shape is ((# in batch) * dim1, dim2, dim3")
dic = {'filename': flatten(batch) for batch in [read_data_file(f) for f in files]}
print(dic['filename'].keys())
print(f"Flattened inputs: {dic['filename']['inputs'].shape}")
print(f"Flattened labels: {dic['filename']['labels'].shape}")


# ===============================================================

def write_test_file(func):
    """Decorator that creates file, runs function, 
    then deletes file and makes sure dump directory is emtpy"""
    def create_file_wraper():
        test_file = write_npz([get_training_example(p) for p in points], "/src/testing/dump")
        print(f'Written file "{test_file}"')
        func()
        print(f'Dumping folder...')
        os.remove('/src/testing/dump/*.npz')
        # Sanity check
        if(len(os.listdir('/src/testing/dump')) !=0):
            print("Issue: file not deleted:")
            for f in os.listdir('/src/testing/dump'):
                print(f)
        
    return create_file_wraper
    
@write_test_file
def test_write():
    print("Running test: Writing file")
    test_file = write_npz([get_training_example(p) for p in points], "/src/testing/dump")
    files = glob(os.path.join('/src/testing/dump', "*.npz"))
    assert(test_file in files)
    assert(len(files) == 1)
    

@write_test_file 
def test_local_read():
    print("Running test: Local read")
    test_file = write_npz([get_training_example(p) for p in points], "/src/testing/dump")
    