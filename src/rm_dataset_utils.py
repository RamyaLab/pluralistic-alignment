import numpy as np
import torch
import os, sys
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

def create_tokenized_ds_a(ds, tokenizer, end_of_conversation_token, max_seq_len):
    # return: ({'input': x, 'inds': [divergence_inds, end_inds]}, y)
        # x: (str(i), 
        # {
        # 'left_input_ids': chosen_input_ids,\
        # 'right_input_ids': rejected_input_ids,\
        # 'left_attention_mask': chosen_attention_mask,\
        # 'right_attention_mask': rejected_attention_mask
        # })
    tokenized_ds = []
    for sample in tqdm(ds, desc='tokenizing dataset'):
        uid = sample['uid']
        prompt = sample['prompt']
        left = sample['left']
        right = sample['right']
        y = sample['y']
        left = prompt + left + end_of_conversation_token
        right = prompt + right + end_of_conversation_token
        left_tokens = tokenizer(left, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
        right_tokens = tokenizer(right, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
        uid = str(uid)
        gen_tokens = {
            'left_input_ids': left_tokens['input_ids'].squeeze(),
            'left_attention_mask': left_tokens['attention_mask'].squeeze(),
            'right_input_ids': right_tokens['input_ids'].squeeze(),
            'right_attention_mask': right_tokens['attention_mask'].squeeze(),
        }
        divergence_id, end_id = get_divergence(gen_tokens['left_input_ids'], gen_tokens['right_input_ids'], 1, max_seq_len, 1)
        new_sample = {
            'input': (uid, gen_tokens),
            'inds': [divergence_id, end_id], 
        }
        tokenized_ds.append((new_sample, y))
    return tokenized_ds

def create_tokenized_ds_b(ds, tokenizer, end_of_conversation_token, max_seq_len):
    # return: ({'input': x, 'inds': [divergence_inds, end_inds]}, y)
    # x: (str(i), 
    # {
    # input_ids: prompt_input_ids,\
    # 'attention_mask': prompt_attention_mask,
    # },
    # {
    # 'left_input_ids': chosen_input_ids,\
    # 'right_input_ids': rejected_input_ids,\
    # 'left_attention_mask': chosen_attention_mask,\
    # 'right_attention_mask': rejected_attention_mask
    # })
    tokenized_ds = []
    for sample in tqdm(ds, desc='tokenizing dataset'):
        uid = sample['uid']
        prompt = sample['prompt']
        left = sample['left']
        right = sample['right']
        y = sample['y']
        prompt += end_of_conversation_token
        left += end_of_conversation_token
        right += end_of_conversation_token
        prompt_tokens = tokenizer(prompt, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
        left_tokens = tokenizer(left, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
        right_tokens = tokenizer(right, truncation=True, padding='max_length', max_length=max_seq_len, return_tensors='pt')
        uid = str(uid)
        prompt_tokens = {
            'input_ids': prompt_tokens['input_ids'].squeeze(),
            'attention_mask': prompt_tokens['attention_mask'].squeeze()
        }
        gen_tokens = {
            'left_input_ids': left_tokens['input_ids'].squeeze(),
            'left_attention_mask': left_tokens['attention_mask'].squeeze(),
            'right_input_ids': right_tokens['input_ids'].squeeze(),
            'right_attention_mask': right_tokens['attention_mask'].squeeze(),
        }
        divergence_id, end_id = get_divergence(gen_tokens['left_input_ids'], gen_tokens['right_input_ids'], 1, max_seq_len, 1)
        new_sample = {
            'input': (uid, prompt_tokens, gen_tokens),
            'inds': [divergence_id, end_id], 
        }
        tokenized_ds.append((new_sample, y))
    return tokenized_ds

def get_divergence(left_token_id, right_token_id, num_padding_at_beginning, seq_len, PAD_ID):
    l_inds = (left_token_id == PAD_ID).nonzero()
    l_ind = l_inds[num_padding_at_beginning].item() if len(l_inds) > num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
    check_divergence = (left_token_id != right_token_id).nonzero().flatten().tolist()
    if len(check_divergence) == 0:
        end_ind = seq_len
        divergence_ind = end_ind - 1
        r_ind = l_ind
    else:
        # Check if there is any padding otherwise take length of sequence
        r_inds = (right_token_id == PAD_ID).nonzero()
        r_ind = r_inds[num_padding_at_beginning].item() if len(r_inds) > num_padding_at_beginning else seq_len
        end_ind = max(l_ind, r_ind)
        divergence_ind = check_divergence[0]
    return divergence_ind, end_ind

def get_bs_divergence(left_token_ids, right_token_ids, bs, num_padding_at_beginning, seq_len, PAD_ID):
    divergence_inds = []
    end_inds = []
    for i in range(bs):
        left_token_id = left_token_ids[i]
        right_token_id = right_token_ids[i]
        divergence_ind, end_ind = get_divergence(left_token_id, right_token_id, num_padding_at_beginning, seq_len, PAD_ID)
        divergence_inds.append(divergence_ind)
        end_inds.append(end_ind)
    return divergence_inds, end_inds
