import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import Literal, Optional

import warnings
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cross_entropy_loss(outputs, targets, reduction='none'):
    # logger.critical(f"outputs: {outputs.shape}, targets: {targets.shape}")
    ce_loss = torch.nn.CrossEntropyLoss(reduction=reduction)
    return ce_loss(outputs, targets)

class SmoothHingeLoss:
    def __init__(self, margin: float=1.0, gamma: float=5.0):
        self.margin = margin
        self.gamma = gamma

    def __call__(self, outputs, targets):
        ty = targets * outputs
        # Case 1: ty >= margin - gamma
        mask_case1 = ty >= (self.margin - self.gamma)
        loss_case1 = 0.5 / self.gamma * torch.clamp(self.margin - ty, min=0) ** 2
        # Case 2: otherwise
        loss_case2 = self.margin - 0.5 * self.gamma - ty
        # Combine losses
        total_loss = torch.where(mask_case1, loss_case1, loss_case2)
        return total_loss


def hinge_loss_smooth(outputs, targets, gamma=5.0):
    '''quadratically smoothed hinge loss'''
    
    ty = targets * outputs
    # Case 1: ty >= 1 - gamma
    mask_case1 = ty >= (1 - gamma)
    loss_case1 = 0.5 / gamma * torch.clamp(1 - ty, min=0) ** 2
    # Case 2: otherwise
    loss_case2 = 1 - 0.5 * gamma - ty
    # Combine losses
    total_loss = torch.where(mask_case1, loss_case1, loss_case2)
    # print(total_loss)
    return total_loss

def hinge_loss_smooth_v1(outputs, targets, gamma=5.0, margin=1):
    """
    Quadratically smoothed hinge loss with adjustable margin.
    """
    ty = targets * outputs
    # Case 1: ty >= margin - gamma
    mask_case1 = ty >= (margin - gamma)
    loss_case1 = 0.5 / gamma * torch.clamp(margin - ty, min=0) ** 2
    # Case 2: otherwise
    loss_case2 = margin - 0.5 * gamma - ty
    # Combine losses
    total_loss = torch.where(mask_case1, loss_case1, loss_case2)
    return total_loss

def hinge_loss_smooth_crazy(outputs, targets, gamma=1.0, margin=1):
    """
    Quadratically smoothed hinge loss with adjustable margin.
    """
    ty = targets * outputs
    # Case 1: ty >= margin - gamma
    mask_case1 = ty >= (margin - gamma)
    loss_case1 = 0.5 / gamma * torch.clamp(margin - ty, min=0) ** 2
    # Case 2: otherwise
    loss_case2 = margin - 0.5 * gamma - ty
    # Combine losses
    total_loss = torch.where(mask_case1, loss_case1, loss_case2)
    return total_loss

def hinge_loss(outputs, targets, margin=1.0):
    # Assume targets are {-1, 1} as typically expected in hinge loss
    hinge_loss_value = margin - targets * outputs
    hinge_loss_value = torch.clamp(hinge_loss_value, min=0)
    return hinge_loss_value

def logistic(z):
    return 1 / (1 + torch.exp(-z))

def nll_logistic_loss(outputs, targets):
    # logger.critical(f"outputs: {outputs.shape}, targets: {targets.shape}")
    # Ensure targets are -1 or 1
    assert torch.all((targets == -1) | (targets == 1)), "Targets must be -1 or 1 for logistic loss."
    probabilities = logistic(targets * outputs)
    return -torch.log(probabilities)

def count_user_samples(train_dataloader):
    user_sample_counts = defaultdict(int)
    tqdmer = tqdm(train_dataloader,desc="scan over the trainset, get user importance")
    for x,_ in tqdmer:  # scan over all samples in training dataset
        u_ids = x[0]
        for u_id in u_ids:
            if u_id not in user_sample_counts.keys():
                user_sample_counts[u_id] = 0 # init a counter for new u_id
            user_sample_counts[u_id] += 1
    return user_sample_counts

class LossFunction:

    def __init__(
        self, 
        loss_type: str,
    ):
        self.loss_type = loss_type
    
    def init_user_importance(self, train_dataloader):
        user_sample_counts = count_user_samples(train_dataloader)
        self.user_importance_dict = defaultdict(int)
        for key in user_sample_counts.keys():
            self.user_importance_dict[key] = 1. / user_sample_counts[key]

    def __call__(self, *args, batch_u_ids=None, **kwargs):
        
        if '_w' in self.loss_type:
            
            if batch_u_ids is None:
                raise ValueError("batch_u_ids is required for weighted loss types but was not provided.")
            if not hasattr(self, 'user_importance_dict') or not self.user_importance_dict:
                raise RuntimeError("user_importance_dict is not initialized. Please ensure scan_ds() is called before using weighted losses.")
            
            batch_user_importances = torch.tensor([self.user_importance_dict[u_id] for u_id in batch_u_ids])
            denominator = batch_user_importances.sum()
            weight_per_sample = batch_user_importances / denominator
        
        if self.loss_type == "hinge":
            loss = hinge_loss(*args, **kwargs)
            return loss.mean()
        elif self.loss_type == "hinge_elementwise":
            loss = hinge_loss(*args, **kwargs)
            return loss
        elif self.loss_type == "logistic":
            loss = nll_logistic_loss(*args, **kwargs)
            return loss.mean()
        elif self.loss_type == "logistic_elementwise":
            loss = nll_logistic_loss(*args, **kwargs)
            return loss
        elif self.loss_type == "CE":
            loss = cross_entropy_loss(*args, **kwargs)
            return loss.mean()
        elif self.loss_type == "CE_elementwise":
            loss = cross_entropy_loss(*args, **kwargs)
            return loss
        elif self.loss_type == "hinge_w":
            loss = hinge_loss(*args, **kwargs)
            loss *= weight_per_sample.to(loss.device)
            return loss.sum()
        elif self.loss_type == "logistic_w":
            loss = nll_logistic_loss(*args, **kwargs)
            loss *= weight_per_sample.to(loss.device)
            return  loss.sum()
        elif self.loss_type == "CE_w":
            loss = cross_entropy_loss(*args, **kwargs)
            loss *= weight_per_sample.to(loss.device)
            return loss.sum()
        elif self.loss_type == "hinge_smooth":
            loss = hinge_loss_smooth(*args, **kwargs)
            return loss.mean()
        elif self.loss_type == "hinge_smooth_v1":
            loss = hinge_loss_smooth_v1(*args, **kwargs)
            return loss.mean()
        elif self.loss_type == "hinge_smooth_crazy":
            loss = hinge_loss_smooth_crazy(*args, **kwargs)
            return loss.mean()
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')


class ProtoBoundLossFunction:
    '''
    bound the magnitude of the protos   0403
    '''
    def __init__(self, upper_bound: float, lmd: float=1):
        self.upper_bound = upper_bound
        self.lmd = lmd
        print(f'the upper bound is {self.upper_bound}')

    def __call__(self, P):
        # the shape of P (k, pref_dims) or the shape of f(Pw) / f(P)
        # print('the items we are bounding:', P.shape)
        return self.lmd * torch.sum(torch.clamp(torch.norm(P,dim=1) - self.upper_bound,min=0))


if __name__ == '__main__':
    
    from .dataset import DummyDataset
    from torch.utils.data import Dataset, DataLoader

    dummy_dataset = DummyDataset(size=100)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True)
    for batch in dummy_dataloader:
        print(batch)
        break
    
    # check the loss_func hinge
    loss_func = LossFunction('hinge')
    loss_func.init_user_importance(dummy_dataloader)
    print(loss_func.user_importance_dict)
    dummy_y_hat = torch.tensor([10,10,10,10]).float()
    dummy_y = torch.tensor([1,1,-1,-1]).float()
    dummy_batch_u_ids = ['1','2','3','2']
    print(loss_func(dummy_y_hat, dummy_y, batch_u_ids=dummy_batch_u_ids))
    
    # check the loss_func hinge_w
    loss_func = LossFunction('hinge_w')
    loss_func.init_user_importance(dummy_dataloader)
    print(loss_func.user_importance_dict)
    dummy_y_hat = torch.tensor([10,10,10,10]).float()
    dummy_y = torch.tensor([1,1,-1,-1]).float()
    dummy_batch_u_ids = ['1','2','3','2']
    print(loss_func(dummy_y_hat, dummy_y, batch_u_ids=dummy_batch_u_ids))
    
    # check the loss_func hinge_w
    loss_func = LossFunction('hinge_smooth')
    loss_func.init_user_importance(dummy_dataloader)
    print(loss_func.user_importance_dict)
    dummy_y_hat = torch.tensor(np.arange(-5,6,0.5)).float()
    dummy_y = torch.tensor(np.ones_like(np.arange(-5,6,0.5))).float()
    dummy_batch_u_ids = ['1'] * len(np.arange(-5,6,0.5))
    print(loss_func(dummy_y_hat, dummy_y, batch_u_ids=dummy_batch_u_ids))
    
    # check the loss_func hinge_w
    loss_func = LossFunction('hinge')
    loss_func.init_user_importance(dummy_dataloader)
    print(loss_func.user_importance_dict)
    dummy_y_hat = torch.tensor(np.arange(-5,6,0.5)).float()
    dummy_y = torch.tensor(np.ones_like(np.arange(-5,6,0.5))).float()
    dummy_batch_u_ids = ['1'] * len(np.arange(-5,6,0.5))
    print(loss_func(dummy_y_hat, dummy_y, batch_u_ids=dummy_batch_u_ids))