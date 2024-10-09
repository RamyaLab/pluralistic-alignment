import numpy as np
import torch
import os
from joblib import dump, load
import uuid

def create_dummy_ds_list(
        size: int = 100,
        seed: int = 42,
        x_dim: int = 768,
        p_dim: int = 4096,
    ):
    rng = np.random.default_rng(seed)
    return [
        (
            (
                str(_id),
                rng.random(p_dim, dtype=np.float32),
                (rng.random(x_dim, dtype=np.float32), rng.random(x_dim, dtype=np.float32))
            ), 
            rng.choice([-1, 1], size=1)[0]
        )
        for _id in range(size)
    ] 

def create_dummy_ds_files(
        size: int = 100,
        seed: int = 42,
        x_dim: int = 768,
        p_dim: int = 4096,
        folder_name: str = './data/dummy_ds/'
    ):
    rng = np.random.default_rng(seed)
    ds_list = [
            (
                (
                    str(_id),
                    rng.random(p_dim, dtype=np.float32),
                    (rng.random(x_dim, dtype=np.float32), rng.random(x_dim, dtype=np.float32))
                ), 
                rng.choice([-1, 1], size=1)[0]
            )
            for _id in range(size)
        ]
    
    for sample in ds_list:
        dump([sample, uuid.uuid4()],os.path.join(folder_name,f'{sample[0][0]}.joblib'))
        print(os.path.join(folder_name,f'{sample[0][0]}.joblib'))

def create_filename(*strings, sep: str = '-'):
    return sep.join(strings)

if __name__ == '__main__':
    create_dummy_ds_files()