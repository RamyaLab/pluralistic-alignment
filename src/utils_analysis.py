from tqdm import tqdm
import numpy as np
import datasets
from datasets import load_dataset
from collections import defaultdict
from scipy.stats import linregress

import os
from omegaconf import OmegaConf
from copy import deepcopy
import torch

from src.datamodule import CustomInstructionDataModule
from src.vanilla_ideal_point_model.lightningmodule import LearnerWrapLightning
from src.utils_ckpt import load_ckpt_learner
from src.utils import create_filename

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_instruction_accu_dict(instruction: list, accu_list: list):
    instru_test_accu_dict = {}
    for i in range(len(instruction)):
        instru_test_accu_dict[instruction[i]] = accu_list[i]
    return instru_test_accu_dict

def calc_overlapping_users(ds:datasets.dataset_dict.DatasetDict):
    train_user_ids = set(ds['train']['user_id'])
    val_user_ids = set(ds['validation']['user_id'])
    test_user_ids = set(ds['test']['user_id'])
    seen_users_val = val_user_ids & train_user_ids
    seen_users_test_1 = test_user_ids & train_user_ids
    seen_users_test_2 = test_user_ids & (train_user_ids | val_user_ids)
    return seen_users_val, seen_users_test_1, seen_users_test_2

def calc_accu_list_per_user(sample_id_user_id_mapping, instruction_accu_dict):
    accu_list_per_user = defaultdict(list)
    num_samples_per_user = defaultdict(int)
    for sample_id, accu in instruction_accu_dict.items():
        user_id = sample_id_user_id_mapping[sample_id]
        accu_list_per_user[user_id].append(accu)
    for user_id, accu_list in accu_list_per_user.items():
        num_samples_per_user[user_id] = len(accu_list)
    avg_accu_per_user = calc_avg_accu_per_user(accu_list_per_user)
    return accu_list_per_user, avg_accu_per_user, num_samples_per_user

def calc_avg_accu_per_user(accu_groupby_users: dict):
    avg_accu_per_user = {}
    for user_id, accu_list in accu_groupby_users.items():
        avg_accu_per_user[user_id] = np.mean(accu_list)
    return avg_accu_per_user

def record_user_id_sample_id_table(ds: datasets.arrow_dataset.Dataset):
    class Recorder:
        def __init__(self):
            self.user_id_sample_id_table = defaultdict(list)
            self.user_id_sample_count_table = defaultdict(int)
        def record_user_id_sample_id_table(self, samples):
            for idx in range(len(samples['ranking_id'])):
                self.user_id_sample_id_table[samples['user_id'][idx]].append(samples['ranking_id'][idx])
                self.user_id_sample_count_table[samples['user_id'][idx]] += 1
    recorder = Recorder()
    ds.map(recorder.record_user_id_sample_id_table, batched=True)
    return recorder.user_id_sample_id_table, recorder.user_id_sample_count_table

def record_rid_uid_table(uid_rid_table: dict) -> dict:
    rid_uid_table = defaultdict(int)
    count = 0
    for uid,rids in uid_rid_table.items():
        count += len(rids)
        for rid in rids:
            rid_uid_table[rid] = uid
    return rid_uid_table

def record_ranking_id_user_id_mapping(ds: datasets.arrow_dataset.Dataset):
    raise DeprecationWarning('This function is deprecated, use record_rid_uid_table instead')
    class Recorder1:
        def __init__(self):
            self.sample_id_user_id_mapping = defaultdict(int)
        def record_user_id_sample_id_table(self, samples):
            # print('here?')
            for idx in range(len(samples['ranking_id'])):
                self.sample_id_user_id_mapping[samples['ranking_id'][idx]] = samples['user_id'][idx]
    recorder = Recorder1()
    ds.map(recorder.record_user_id_sample_id_table, batched=True)
    return recorder.sample_id_user_id_mapping

def filter_users_by_sample_count(user_sample_count_dict: dict, min_sample_count:int):
    filtered_user_ids = {user_id:sample_count for user_id, sample_count in user_sample_count_dict.items()\
        if sample_count >= min_sample_count}
    return filtered_user_ids

def calc_linear_reg(x:list,y:list):
    result = linregress(x,y)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = result.slope * x_line + result.intercept
    print(f"Slope: {result.slope}")
    print(f"Intercept: {result.intercept}")
    print(f"R-squared: {result.rvalue**2}")
    print(f"P-value: {result.pvalue}")
    print(f"Standard error of the slope: {result.stderr}")
    return x_line, y_line

def infer_accu(learner, dataloader):
    accu_list = []
    original_device = learner.device
    learner.to('cpu')
    learner.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader,desc='infer_accu'):
            x,y = batch
            # batch_u_ids = x[0]
            y_hat = learner.preference_learner(x)
            if learner.preference_learner_params.pref_learner_type in ['dist','dist_normalization','norm', 'angle_hinge']:
                accu = (((y_hat * y) > 0).to(torch.float)).detach().cpu().numpy()
            elif learner.preference_learner_params.pref_learner_type in ['angle', 'dist_logistic']:
                y = (y+1)/2
                y = y.to(torch.long)
                # fix-patch for an issue in old simulated dataset {-1,3}
                if y.max() > 1:
                    y = (y / 2).to(torch.long)
                accu = (y_hat.argmax(dim=1) == y).detach().cpu().numpy()
            accu_list.extend(list(accu))
    learner.to(original_device)
    learner.train() 
    # we need to manually call train(), since this function will be called by 
    # on_train_epoch_end(), which will not automatically switch the model to train mode.
    return accu_list

def infer_fx_fu(learner, dataloader, num_samples=10000):
    fx0_record = []
    fx1_record = []
    fu_record = []
    count = 0
    learner.eval()
    tqdmr = tqdm(dataloader, desc='infering fx and fu')
    for x,y in tqdmr:
        u_ids, p, (x_left, x_right) = x
        x_left_prime, x_right_prime = learner.preference_learner.item_learner((p, x_left, x_right))
        u_prime = learner.preference_learner.user_learner(u_ids)
        fx0_record.append(x_left_prime)
        fx1_record.append(x_right_prime)
        fu_record.append(u_prime)
        count += len(y)
        tqdmr.set_postfix({'count':count})
        if count >= num_samples and num_samples != -1:
            break
    fx0_record = torch.vstack(fx0_record).detach().cpu().numpy()
    fx1_record = torch.vstack(fx1_record).detach().cpu().numpy()
    fu_record = torch.vstack(fu_record).detach().cpu().numpy()
    return fx0_record, fx1_record, fu_record

def infer_fx_fu_moe(learner, dataloader, num_samples=10000):
    fx0_record = []
    fx1_record = []
    fu_record = []
    count = 0
    learner.eval()
    tqdmr = tqdm(dataloader, desc='infering fx and fu')
    for x,y in tqdmr:
        u_ids, p, (x_left, x_right) = x
        x_left_prime, x_right_prime, u_prime = learner.preference_learner.map_to_pref_embedding_space(x)
        fx0_record.append(x_left_prime)
        fx1_record.append(x_right_prime)
        fu_record.append(u_prime)
        count += len(y)
        tqdmr.set_postfix({'count':count})
        if count >= num_samples and num_samples != -1:
            break
    fx0_record = torch.vstack(fx0_record).detach().cpu().numpy()
    fx1_record = torch.vstack(fx1_record).detach().cpu().numpy()
    fu_record = torch.vstack(fu_record).detach().cpu().numpy()
    return fx0_record, fx1_record, fu_record

def get_original_latents(dataloader):
    
    prompt_latents = []
    img0_latents = []
    img1_latents = []
    for x,y in tqdm(dataloader):
        prompt_latents.append(x[1].detach().cpu().numpy())
        img0_latents.append(x[2][0].detach().cpu().numpy())
        img1_latents.append(x[2][1].detach().cpu().numpy())

    prompt_latents = np.vstack(prompt_latents)
    img0_latents = np.vstack(img0_latents)
    img1_latents = np.vstack(img1_latents)

    prompt_img0_latents = np.hstack([prompt_latents, img0_latents])
    prompt_img1_latents = np.hstack([prompt_latents, img1_latents])

    all_combination_latents = np.vstack([prompt_img0_latents, prompt_img1_latents])
    all_combination_latents_unique = np.unique(all_combination_latents,axis=0)
    
    return all_combination_latents, all_combination_latents_unique

def infer_accu_original_latents(data_loader):
    is_accu_list = []
    for x, y in data_loader:
        user_ids, prompts, (x_lefts, x_rights) = x
        is_left_larger = torch.diag(prompts @ x_lefts.T) >= torch.diag(prompts @ x_rights.T)
        y_hat = -(is_left_larger * 2 - 1)
        is_accu_list.extend((y_hat == y).tolist())
    return is_accu_list