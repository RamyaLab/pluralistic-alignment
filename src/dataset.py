from torch.utils.data import Dataset
from typing import Sequence, List, Tuple
from joblib import load
import torch
import os
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor   # 2:24 streaming = False (baseline 5mins)
from multiprocessing import Pool    # 6:48 streaming = False (baseline 5mins)

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DummyDataset(Dataset):
    def __init__(self, size, dim=1024):
        """
        Initialize the dummy dataset.
        Args:
        - size (int): The size of the dataset.
        """
        self.size = size
        self.dim = dim

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return self.size

    def __getitem__(self, idx):
        """
        Generate a sample of the data.
        """
        u_id = str(idx%10)  # User ID as a string of int
        p = torch.randn(self.dim)  # A tensor for p
        x_left = torch.randn(self.dim)  # A tensor for x_left
        x_right = torch.randn(self.dim)  # A tensor for x_right
        y = random.choice([-1,1])
        return (u_id, p, (x_left, x_right)), y

class PreferenceDataset(Dataset):

    ds_list: list

    def __init__(self, ds_list: list):
        self.ds_list = ds_list

    def __len__(self):
        return len(self.ds_list)

    def __getitem__(self, index):
        return self.ds_list[index]

class CustomInstructionDataset(Dataset):
    
    def __init__(self, data_instr, ds_path, streaming=True):
        self.ds_path = ds_path  # the path to the folder which contains all samples
        if type(data_instr) == str:
            self.data_instructions = torch.load(data_instr)  # [unique_ids,...]
        elif type(data_instr) == list:
            self.data_instructions = data_instr
        else:
            raise ValueError('data_instr should be either a string or a list')
        self.ds_file_path = [
            os.path.join(self.ds_path, str(unique_id) + '.joblib')
            for unique_id in self.data_instructions
        ]
        self.streaming = streaming
        if not streaming:
            with ThreadPoolExecutor(max_workers=16) as executor:
                tqdmr = tqdm(total=len(self.ds_file_path), desc='Loading all samples into RAM')
                # Map load function to each file path and use tqdm to show progress
                futures = [executor.submit(load, path) for path in self.ds_file_path]
                self.ds_list = [future.result() for future in tqdm(futures, total=len(futures))]
        logger.critical(f"Dataset loaded: {len(self)} samples")
        
    def __len__(self):
        return len(self.data_instructions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.streaming:
            sample = load(self.ds_file_path[idx])
        else:
            sample = self.ds_list[idx]
        # NOTICE: sample[1] stores the sample unique id (omit it)
        return sample[0]    # (u_id, p, (x_left, x_right)), y
    
class CustomInstructionDataset_llm(Dataset):
    
    def __init__(self, data_instr, ds_path, streaming=True):
        self.ds_path = ds_path  # the path to the folder which contains all samples
        if type(data_instr) == str:
            self.data_instructions = torch.load(data_instr)  # [unique_ids,...]
        elif type(data_instr) == list:
            self.data_instructions = data_instr
        else:
            raise ValueError('data_instr should be either a string or a list')
        self.ds_file_path = [
            os.path.join(self.ds_path, str(unique_id) + '.joblib')
            for unique_id in self.data_instructions
        ]
        self.streaming = streaming
        if not streaming:
            with ThreadPoolExecutor(max_workers=16) as executor:
                tqdmr = tqdm(total=len(self.ds_file_path), desc='Loading all samples into RAM')
                # Map load function to each file path and use tqdm to show progress
                futures = [executor.submit(load, path) for path in self.ds_file_path]
                self.ds_list = [future.result() for future in tqdm(futures, total=len(futures))]
        logger.critical(f"Dataset loaded: {len(self)} samples")
        
    def __len__(self):
        return len(self.data_instructions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.streaming:
            sample = load(self.ds_file_path[idx])
        else:
            sample = self.ds_list[idx]
        return sample   # ({'input':x, 'inds': [divergence_inds, end_inds]}, y)
                        # x = (uid, (left_gen, right_gen))

class SummaryDataset(Dataset):
    
    def __init__(self, data_instr, ds_path, prompt_embeds_path, summary_embeds_path) -> None:
        super().__init__()
        self.ds_path = ds_path  # the path to the folder which contains all samples
        if type(data_instr) == str:
            self.ds_file_path = self.data_instructions = torch.load(data_instr)  # [unique_ids,...]
        elif type(data_instr) == list:
            self.data_instructions = data_instr
        else:
            raise ValueError('data_instr should be either a string or a list')
        self.prompt_embedding = torch.load(prompt_embeds_path)
        self.summary_embedding = torch.load(summary_embeds_path)
        logger.critical(f"Dataset loaded: {len(self)} samples")

    def __len__(self):
        return len(self.data_instructions)
    
    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.load(self.ds_file_path[idx])
        sample = (sample[0][0], self.prompt_embedding[sample[0][1]], (self.summary_embedding[sample[0][2][0]], self.summary_embedding[sample[0][2][1]])), sample[1]
        return sample    # (u_id, p, (x_left, x_right)), y

if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf_seen_ds = OmegaConf.load('./config_ds/pickapicv2_cliph_seen_user_ds.yaml')
    ds = CustomInstructionDataset(
        data_instr_file_path = conf_seen_ds.train_instr_file_path,
        ds_path=conf_seen_ds.ds_path,
        streaming=False,
    )
    print(ds[0])
