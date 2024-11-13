# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

'''
part of the code was adopted from microsoft/DeepSpeed
'''

import torch
import os
# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, Subset, ConcatDataset
from transformers import AutoTokenizer

import re
import numpy as np

import logging
logger = logging.getLogger(__name__)


def preprocess_tokenized_ds(batch, device, num_padding_at_beginning=1, seq_len=512, PAD_ID=1):
    size = int(batch['input_ids'].shape[0] / 2)
    divergence_inds, end_inds = get_bs_divergence(batch['input_ids'], size, num_padding_at_beginning, seq_len, PAD_ID)
    batch = to_device(batch, device)
    chosen_input_ids, rejected_input_ids = batch['input_ids'][:size], batch['input_ids'][size:]
    chosen_attention_mask, rejected_attention_mask = batch['attention_mask'][:size], batch['attention_mask'][size:]
    # FIXME: not sure whether need to randomize the order of the chosen and rejected
    x = {
        'left_input_ids': chosen_input_ids,
        'right_input_ids': rejected_input_ids,
        'left_attention_mask': chosen_attention_mask,
        'right_attention_mask': rejected_attention_mask,
    }
    x = ([str(0)]*size, x) # since the dataset has no user id, we only use one user id for now
    samples = ({'input':x, 'inds':[divergence_inds, end_inds]}, [-1]*size)
    return samples

def preprocess_tokenized_ds_rm_pal_b(batch, device, num_padding_at_beginning=1, seq_len=512, PAD_ID=1):
    # logger.critical(f"batch: {batch}")
    prompt_token_ids = batch[0][:,0,:]
    chosen_token_ids, rejected_token_ids= batch[2][:,0,:], batch[4][:,0,:]
    size = prompt_token_ids.size(0)
    divergence_inds, end_inds = get_bs_divergence_rm_pal_b(chosen_token_ids, rejected_token_ids, size, num_padding_at_beginning, seq_len, PAD_ID)
    batch = to_device(batch, device)
    prompt_token_ids, prompt_attention_mask = batch[0][:,0,:], batch[1][:,0,:]
    chosen_token_ids, chosen_attention_mask = batch[2][:,0,:], batch[3][:,0,:]
    rejected_token_ids, rejected_attention_mask = batch[4][:,0,:], batch[5][:,0,:]
    # FIXME: not sure whether need to randomize the order of the chosen and rejected
    p_x = {'input_ids': prompt_token_ids, 'attention_mask': prompt_attention_mask}
    x = {
        'left_input_ids': chosen_token_ids,
        'right_input_ids': rejected_token_ids,
        'left_attention_mask': chosen_attention_mask,
        'right_attention_mask': rejected_attention_mask,
    }
    x = ([str(0)]*size, p_x, x)
    samples = ({'input': x, 'inds': [divergence_inds, end_inds]}, [-1]*size)
    return samples



def get_tokenized_ds_rm_pal_b(
    ds_name: str,
    ds_output_path: str,
    data_split: str,
    split_index: int,
    train_phase: int,
    seed: int,
    tokenizer: AutoTokenizer,
):
    train_ds_token_ids_path = f'train_ds_rm_pal_b_{ds_name.replace('/', '-')}_seed{seed}_{data_split}_{split_index}.pt'
    val_ds_token_ids_path = f'val_ds_rm_pal_b_{ds_name.replace('/', '-')}_seed{seed}_{data_split}_{split_index}.pt'
    test_ds_token_ids_path = f'test_ds_rm_pal_b_{ds_name.replace('/', '-')}_seed{seed}_{data_split}_{split_index}.pt'

    if not os.path.exists(os.path.join(ds_output_path, train_ds_token_ids_path)) \
        or not os.path.exists(os.path.join(ds_output_path, val_ds_token_ids_path)) \
        or not os.path.exists(os.path.join(ds_output_path, test_ds_token_ids_path)):
        print(f"Tokenized dataset not found at {train_ds_token_ids_path}")
        print(f"Tokenized dataset not found at {val_ds_token_ids_path}")
        print(f"Tokenized dataset not found at {test_ds_token_ids_path}")
        print("Tokenizing the dataset...")
        raw_ds = get_raw_dataset(dataset_name=ds_name, output_path=ds_output_path, seed=seed)
        train_ds = raw_ds.get_train_data()
        split_ids = get_raw_dataset_split_index(
            output_path=ds_output_path,
            dataset_name=ds_name,
            seed=seed,
            split_name='train',
            data_split=data_split,
            split_index=split_index,
            data_size=len(train_ds),
            rebuild=False,
        )
        train_ids, val_ids = split_ids[:int(len(split_ids)*0.9)], split_ids[int(len(split_ids)*0.9):]
        train_ds, val_ds = Subset(train_ds, train_ids), Subset(train_ds, val_ids)

        train_ds = create_dataset_split_rm_pal_b(train_ds, raw_ds, train_phase, tokenizer, 
                                        end_of_conversation_token="<|endoftext|>",
                                        max_seq_len=512)
        val_ds = create_dataset_split_rm_pal_b(val_ds, raw_ds, train_phase, tokenizer,
                                        end_of_conversation_token="<|endoftext|>",
                                        max_seq_len=512)

        test_ds = raw_ds.get_eval_data()
        split_ids = get_raw_dataset_split_index(
            output_path=ds_output_path,
            dataset_name=ds_name,
            seed=seed,
            split_name='eval',
            data_split=data_split,
            split_index=split_index,
            data_size=len(test_ds),
            rebuild=False,
        )
        test_ds = Subset(test_ds, split_ids)
        test_ds = create_dataset_split_rm_pal_b(test_ds, raw_ds, train_phase, tokenizer, 
                                    end_of_conversation_token="<|endoftext|>",
                                    max_seq_len=512)

        torch.save(train_ds, os.path.join(ds_output_path, train_ds_token_ids_path))
        torch.save(val_ds, os.path.join(ds_output_path, val_ds_token_ids_path))
        torch.save(test_ds, os.path.join(ds_output_path, test_ds_token_ids_path))
        print(f"Tokenized dataset saved at {train_ds_token_ids_path}")
        print(f"Tokenized dataset saved at {val_ds_token_ids_path}")
        print(f"Tokenized dataset saved at {test_ds_token_ids_path}")

    train_ds = torch.load(os.path.join(ds_output_path, train_ds_token_ids_path))
    val_ds = torch.load(os.path.join(ds_output_path, val_ds_token_ids_path))
    test_ds = torch.load(os.path.join(ds_output_path, test_ds_token_ids_path))
    
    return train_ds, val_ds, test_ds

def get_tokenized_ds(
    ds_name: str,
    ds_output_path: str,
    data_split: str,
    split_index: int,
    train_phase: int,
    seed: int,
    tokenizer: AutoTokenizer,
    **kwargs,
):
    train_ds_token_ids_path = f'train_ds_{ds_name.replace('/', '-')}_seed{seed}_{data_split}_{split_index}.pt'
    val_ds_token_ids_path = f'val_ds_{ds_name.replace('/', '-')}_seed{seed}_{data_split}_{split_index}.pt'
    test_ds_token_ids_path = f'test_ds_{ds_name.replace('/', '-')}_seed{seed}_{data_split}_{split_index}.pt'

    if not os.path.exists(os.path.join(ds_output_path, train_ds_token_ids_path)) \
        or not os.path.exists(os.path.join(ds_output_path, val_ds_token_ids_path)) \
        or not os.path.exists(os.path.join(ds_output_path, test_ds_token_ids_path)):
        logger.warning(f"Tokenized dataset not found at {train_ds_token_ids_path}")
        logger.warning(f"Tokenized dataset not found at {val_ds_token_ids_path}")
        logger.warning(f"Tokenized dataset not found at {test_ds_token_ids_path}")
        logger.warning("Tokenizing the dataset...")
        raw_ds = get_raw_dataset(dataset_name=ds_name, output_path=ds_output_path, seed=seed)
        train_ds = raw_ds.get_train_data()
        split_ids = get_raw_dataset_split_index(
            output_path=ds_output_path,
            dataset_name=ds_name,
            seed=seed,
            split_name='train',
            data_split=data_split,
            split_index=split_index,
            data_size=len(train_ds),
            rebuild=False,
        )
        train_ids, val_ids = split_ids[:int(len(split_ids)*0.9)], split_ids[int(len(split_ids)*0.9):]
        train_ds, val_ds = Subset(train_ds, train_ids), Subset(train_ds, val_ids)

        train_ds = create_dataset_split(train_ds, raw_ds, train_phase, tokenizer, 
                                        end_of_conversation_token="<|endoftext|>",
                                        max_seq_len=512)
        val_ds = create_dataset_split(val_ds, raw_ds, train_phase, tokenizer,
                                        end_of_conversation_token="<|endoftext|>",
                                        max_seq_len=512)

        test_ds = raw_ds.get_eval_data()
        split_ids = get_raw_dataset_split_index(
            output_path=ds_output_path,
            dataset_name=ds_name,
            seed=seed,
            split_name='eval',
            data_split=data_split,
            split_index=split_index,
            data_size=len(test_ds),
            rebuild=False,
        )
        test_ds = Subset(test_ds, split_ids)
        test_ds = create_dataset_split(test_ds, raw_ds, train_phase, tokenizer, 
                                    end_of_conversation_token="<|endoftext|>",
                                    max_seq_len=512)

        torch.save(train_ds, os.path.join(ds_output_path, train_ds_token_ids_path))
        torch.save(val_ds, os.path.join(ds_output_path, val_ds_token_ids_path))
        torch.save(test_ds, os.path.join(ds_output_path, test_ds_token_ids_path))
        logger.warning(f"Tokenized dataset saved at {train_ds_token_ids_path}")
        logger.warning(f"Tokenized dataset saved at {val_ds_token_ids_path}")
        logger.warning(f"Tokenized dataset saved at {test_ds_token_ids_path}")

    train_ds = torch.load(os.path.join(ds_output_path, train_ds_token_ids_path))
    val_ds = torch.load(os.path.join(ds_output_path, val_ds_token_ids_path))
    test_ds = torch.load(os.path.join(ds_output_path, test_ds_token_ids_path))
    
    return train_ds, val_ds, test_ds


def to_device(batch, device):
    try:
        output = {}
        for k, v in batch.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
    except AttributeError:
        output = []
        for v in batch:
            try:
                output.append(v.to(device))
            except:
                logger.warning(f"Failed to move {v} to device")
                output.append(v)
    return output

def convert_ids_to_string(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def get_bs_divergence(input_ids, bs, num_padding_at_beginning, seq_len, PAD_ID):
    divergence_inds = []
    end_inds = []
    chosen_ids = input_ids[:bs]  # bs x seq
    rejected_ids = input_ids[bs:]
    for i in range(bs):
        chosen_id = chosen_ids[i]
        rejected_id = rejected_ids[i]
        c_inds = (chosen_id == PAD_ID).nonzero()
        c_ind = c_inds[num_padding_at_beginning].item() if len(c_inds) > num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
        check_divergence = (chosen_id != rejected_id).nonzero().flatten().tolist()
        if len(check_divergence) == 0:
            end_ind = seq_len
            divergence_ind = end_ind - 1
            r_ind = c_ind
        else:
            # Check if there is any padding otherwise take length of sequence
            r_inds = (rejected_id == PAD_ID).nonzero()
            r_ind = r_inds[num_padding_at_beginning].item(
            ) if len(r_inds) > num_padding_at_beginning else seq_len
            end_ind = max(c_ind, r_ind)
            divergence_ind = check_divergence[0]
        divergence_inds.append(divergence_ind)
        end_inds.append(end_ind)
    return divergence_inds, end_inds

def get_bs_divergence_rm_pal_b(chosen_ids, rejected_ids, bs, num_padding_at_beginning, seq_len, PAD_ID):
    divergence_inds = []
    end_inds = []
    for i in range(bs):
        chosen_id = chosen_ids[i]
        rejected_id = rejected_ids[i]
        c_inds = (chosen_id == PAD_ID).nonzero()
        c_ind = c_inds[num_padding_at_beginning].item() if len(c_inds) > num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
        check_divergence = (chosen_id != rejected_id).nonzero().flatten().tolist()
        if len(check_divergence) == 0:
            end_ind = seq_len
            divergence_ind = end_ind - 1
            r_ind = c_ind
        else:
            # Check if there is any padding otherwise take length of sequence
            r_inds = (rejected_id == PAD_ID).nonzero()
            r_ind = r_inds[num_padding_at_beginning].item(
            ) if len(r_inds) > num_padding_at_beginning else seq_len
            end_ind = max(c_ind, r_ind)
            divergence_ind = check_divergence[0]
        divergence_inds.append(divergence_ind)
        end_inds.append(end_ind)
    return divergence_inds, end_inds

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat(
            [f[0] for f in data] + [f[2] for f in data], dim=0)
        batch["attention_mask"] = torch.cat(
            [f[1] for f in data] + [f[3] for f in data], dim=0)
        return batch
    
class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id

class PromptDataset_rm_pal_b(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id

def create_dataset_split_rm_pal_b(current_dataset, raw_dataset, train_phase, tokenizer, end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    for i, tmp_data in enumerate(current_dataset):
        # tokenize the text
        prompt_sentence = raw_dataset.get_prompt(tmp_data)
        chosen_sentence = raw_dataset.get_chosen(tmp_data)  # the accept response
        reject_sentence = raw_dataset.get_rejected( tmp_data)  # the accept response
        if chosen_sentence is not None and reject_sentence is not None:
            prompt_sentence += end_of_conversation_token
            chosen_sentence += end_of_conversation_token  # the accept response
            reject_sentence += end_of_conversation_token
            prompt_token = tokenizer(prompt_sentence,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
            chosen_token = tokenizer(chosen_sentence,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
            reject_token = tokenizer(reject_sentence,
                                        max_length=max_seq_len,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")
            prompt_dataset.append(prompt_token)
            chosen_dataset.append(chosen_token)
            reject_dataset.append(reject_token)
    print(
        f'Creating dataset {raw_dataset.dataset_name_clean} size={len(chosen_dataset)}'
    )
    return PromptDataset_rm_pal_b(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)

def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)
        print(
            f'Creating dataset {raw_dataset.dataset_name_clean} for {train_phase=} size={len(chosen_dataset)}'
        )

    elif train_phase == 3:
        filtered = 0
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                if prompt_token["input_ids"].size()[-1] <= max_seq_len:
                    for key_word in ["input_ids", "attention_mask"]:
                        prompt_token[key_word] = prompt_token[
                            key_word].squeeze(0).flip(0)
                    prompt_dataset.append(prompt_token)
                else:
                    filtered += 1
        print(f'Creating dataset {raw_dataset.dataset_name_clean} '
              f'for {train_phase=} size={len(prompt_dataset)} {filtered=}')

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(
    output_path,
    dataset_name,
    seed,
    split_name,
    data_split,
    split_index,
    data_size,
    rebuild=False
):
    index_file_name = f"{output_path}/{dataset_name}/seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if not os.path.exists(f"{output_path}/{dataset_name}"):
        os.makedirs(f"{output_path}/{dataset_name}")
    # reindex each time when using local jsonfile since it's more likely to get modified
    if rebuild or (not os.path.isfile(index_file_name)) or (dataset_name
                                                            == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}/seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, dataset_name):
        self.output_path = output_path
        self.seed = seed
        if os.path.exists(dataset_name):
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['prompt'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['chosen']

    def get_rejected(self, sample):
        return " " + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample[
            'rejected']


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt'] + "Assistant:"

    def get_chosen(self, sample):
        return sample['chosen'].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['question']['full_text'] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['history'] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample['history'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample['history'] + " Assistant: " + response


# English dataset
class PvduySharegptalpacaoavicunaformatDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "pvduy/sharegpt_alpaca_oa_vicuna_format"
        self.dataset_name_clean = "pvduy_sharegpt_alpaca_oa_vicuna_format"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        if sample['prompt'] is not None and len(sample['prompt']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant")
        return None

    def get_chosen(self, sample):
        if sample['label'] is not None and len(sample['label']) > 0:
            return " " + sample['label']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['label'] is not None and len(
                sample['prompt']) > 0 and len(sample['label']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant") + " " + sample['label']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name, chat_path):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             chat_path + '/data/train.json',
                                             "eval":
                                             chat_path + '/data/eval.json'
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return " " + sample['prompt']
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        if sample['chosen'] is not None:
            return " " + sample['chosen']
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        if sample['rejected'] is not None:
            return " " + sample['rejected']
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['chosen'] is not None:
            return " " + sample['prompt'] + " " + sample['chosen']
        return None

    def get_prompt_and_rejected(self, sample):
        if sample['prompt'] is not None and sample['rejected'] is not None:
            return " " + sample['prompt'] + " " + sample['rejected']
        return None


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "wangrui6/Zhihu-KOL"
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return " Human: " + sample['INSTRUCTION'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return " " + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return " Human: " + sample[
                'INSTRUCTION'] + " Assistant: " + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "Cohere/miracl-zh-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_zh_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'negative_passages'][0]['text']


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        self.dataset_name_clean = "Hello_SimpleAI_HC3_Chinese"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['question'] is not None:
            return " Human: " + sample['question'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['human_answers'][0] is not None:
            return " " + sample['human_answers'][0]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['question'] is not None and sample['human_answers'][
                0] is not None:
            return " Human: " + sample['question'] + " Assistant: " + sample[
                'human_answers'][0]
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "mkqa-Chinese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['zh_cn'] is not None:
            return " Human: " + sample['queries']['zh_cn'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['zh_cn'][0]['text'] is not None:
            return " " + sample['answers']['zh_cn'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['zh_cn'] is not None and sample['answers'][
                'zh_cn'][0]['text'] is not None:
            return " Human: " + sample['queries'][
                'zh_cn'] + " Assistant: " + sample['answers']['zh_cn'][0][
                    'text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "mkqa-Japanese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['ja'] is not None:
            return " Human: " + sample['queries']['ja'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['ja'][0]['text'] is not None:
            return " " + sample['answers']['ja'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['ja'] is not None and sample['answers']['ja'][0][
                'text'] is not None:
            return " Human: " + sample['queries'][
                'ja'] + " Assistant: " + sample['answers']['ja'][0]['text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "Cohere/miracl-ja-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_ja_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        if len(sample['negative_passages']) > 0:
            return " Human: " + sample['query'] + " Assistant: " + sample[
                'negative_passages'][0]['text']
        return None


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "lmqg/qg_jaquad"
        self.dataset_name_clean = "lmqg_qg_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['question'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['sentence']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['question'] + " Assistant: " + sample[
            'sentence']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, dataset_name):
        super().__init__(output_path, seed, dataset_name)
        self.dataset_name = "lmqg/qag_jaquad"
        self.dataset_name_clean = "lmqg_qag_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['paragraph']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant: " + sample[
            'paragraph']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


def get_raw_dataset(dataset_name, output_path, seed):

    if "Dahoas/rm-static" in dataset_name:
        return DahoasRmstaticDataset(output_path, seed, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return DahoasFullhhrlhfDataset(output_path, seed, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return OpenaiWebgptcomparisonsDataset(
            output_path, seed, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return StanfordnlpSHPDataset(output_path, seed, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return Wangrui6ZhihuKOLDataset(output_path, seed, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return CohereMiraclzhqueries2212Dataset(
            output_path, seed, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return HelloSimpleAIHC3ChineseDataset(
            output_path, seed, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return MkqaChineseDataset(output_path, seed,
                                               "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return MkqaJapaneseDataset(output_path, seed,
                                                "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return CohereMiracljaqueries2212Dataset(
            output_path, seed, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return LmqgQgjaquadDataset(output_path, seed,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return LmqgQagjaquadDataset(output_path, seed,
                                                 dataset_name)
    elif "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir,
                         os.path.pardir, os.path.pardir))
        if not (os.path.isfile(chat_path + '/data/train.json')
                and os.path.isfile(chat_path + '/data/eval.json')):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return LocalJsonFileDataset(output_path, seed,
                                                 dataset_name, chat_path)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in llm_dataset.py."
        )