from sklearn.model_selection import train_test_split
from typing import Literal
import os, torch

import logging
logger = logging.getLogger(__name__)

def dataset_factory(dataset_name: str, ds_output_path, model_type: Literal['b', 'a'], end_of_conversation_token: str, max_seq_len: int, tokenizer, **kwargs):
    
    # logger.critical(f" 💠 {kwargs}")
    if model_type == 'a':
        from .rm_dataset_utils import create_tokenized_ds_a as create_tokenized_ds
    elif model_type == 'b':
        from .rm_dataset_utils import create_tokenized_ds_b as create_tokenized_ds
    
    ds_output_path = os.path.join(ds_output_path, dataset_name)
    if 'num_samples_per_unseen_train' not in kwargs:
        train_ds_path = os.path.join(ds_output_path, f'{tokenizer.name_or_path.split('/')[-1]}_train_ds.pt')
        val_ds_path = os.path.join(ds_output_path, f'{tokenizer.name_or_path.split('/')[-1]}_val_ds.pt')
        test_ds_path = os.path.join(ds_output_path, f'{tokenizer.name_or_path.split('/')[-1]}_test_ds.pt')
        # logger.critical(f" 💠 {train_ds_path}")
        # logger.critical(f" 💠 {val_ds_path}")
        # logger.critical(f" 💠 {test_ds_path}")
    else:
        train_ds_path = os.path.join(ds_output_path, f'{tokenizer.name_or_path.split('/')[-1]}_{kwargs['num_samples_per_unseen_train']}_train_ds.pt')
        val_ds_path = os.path.join(ds_output_path, f'{tokenizer.name_or_path.split('/')[-1]}_{kwargs['num_samples_per_unseen_train']}_val_ds.pt')
        test_ds_path = os.path.join(ds_output_path, f'{tokenizer.name_or_path.split('/')[-1]}_{kwargs['num_samples_per_unseen_train']}_test_ds.pt')
        # logger.critical(f" 💠 {train_ds_path}")
        # logger.critical(f" 💠 {val_ds_path}")
        # logger.critical(f" 💠 {test_ds_path}")

    if os.path.exists(train_ds_path) and os.path.exists(val_ds_path) and os.path.exists(test_ds_path):
        logger.critical(f" 💠 Loading tokenized datasets from {ds_output_path}")
        train_ds = torch.load(train_ds_path)
        val_ds = torch.load(val_ds_path)
        test_ds = torch.load(test_ds_path)
        return train_ds, val_ds, test_ds
    else:
        logger.critical(' 💠 Tokenized datasets not found, creating tokenized datasets...')
        os.makedirs(ds_output_path, exist_ok=True)
    
    if dataset_name == 'openai/summarize_from_feedback':
        from .summary_dataset_utils import preprocess_ds
        logger.critical(kwargs)
        train_ds, test_ds = preprocess_ds(**kwargs)
        tokenized_train_ds = create_tokenized_ds(train_ds, tokenizer, end_of_conversation_token, max_seq_len)
        tokenized_test_ds = create_tokenized_ds(test_ds, tokenizer, end_of_conversation_token, max_seq_len)
        tokenized_train_ds, tokenized_val_ds = train_test_split(tokenized_train_ds, test_size=0.1, random_state=42, stratify=[x[0]['input'][0] for x in tokenized_train_ds])
        torch.save(tokenized_train_ds, train_ds_path)
        torch.save(tokenized_val_ds, val_ds_path)
        torch.save(tokenized_test_ds, test_ds_path)
        return tokenized_train_ds, tokenized_val_ds, tokenized_test_ds
    if dataset_name == 'openai/summarize_from_feedback_unseen':
        from .summary_dataset_utils import preprocess_unseen_ds
        logger.critical(kwargs)
        train_ds, test_ds = preprocess_unseen_ds(**kwargs)
        tokenized_train_ds = create_tokenized_ds(train_ds, tokenizer, end_of_conversation_token, max_seq_len)
        tokenized_test_ds = create_tokenized_ds(test_ds, tokenizer, end_of_conversation_token, max_seq_len)
        # tokenized_train_ds, tokenized_val_ds = train_test_split(tokenized_train_ds, test_size=0.1, random_state=42, stratify=[x[0]['input'][0] for x in tokenized_train_ds])
        tokenized_val_ds = tokenized_test_ds # when sample size is too small, the train_test_split will fail, we direct use the test set as validation set
        torch.save(tokenized_train_ds, train_ds_path)
        torch.save(tokenized_val_ds, val_ds_path)
        torch.save(tokenized_test_ds, test_ds_path)
        return tokenized_train_ds, tokenized_val_ds, tokenized_test_ds
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet, please add the dataset to the dataset_factory.py file")
    
    
