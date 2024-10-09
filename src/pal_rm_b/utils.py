import numpy as np
import torch

from typing import Literal
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calc_cumulative(loss):
    l = len(loss)
    if l == 0:
        return 0.0  # Handle empty list case
    # weighted sum
    cumulative_loss = 0.0
    for i in range(l):
        cumulative_loss += loss[i] * (i + 1)
    # normalization
    cumulative_loss /= l * (l + 1) / 2
    return cumulative_loss

def calc_mean(loss):
    l = len(loss)
    if l == 0:
        return 0.0  # Handle empty list case
    return sum(loss) / l


def calc_only_last(loss):
    l = len(loss)
    if l == 0:
        return 0.0  # Handle empty list case
    return loss[-1]

def calc_hinge_loss(y_hat, y, loss_fn, inds, weightingMethod: Literal['mean', 'cumulative', 'only_last']='mean'):
    '''
    this function calculate the pairwise loss given the predicted reward score and the true choice
    y_hat: (bs, max_token_length), NOTICE: when we use hinge loss, y_hat represents the reward difference between two responses
    y: (bs,)
    inds: [divergence_inds, end_inds]
    '''
    bs = y_hat.size(0)
    loss = 0.
    accu_all_tokens = 0
    accu_last_token = 0
    for i in range(bs):
        divergence_ind, end_ind = inds[0][i], inds[1][i]
        # logger.critical(f"divergence_ind: {divergence_ind}, end_ind: {end_ind}")
        y_hat_i = y_hat[i]  # (max_token_length,)
        # logger.critical(f"y_hat_i: {y_hat_i.shape}")
        y_hat_truncated = y_hat_i[divergence_ind:end_ind]
        y_truncated = y[i] * torch.ones_like(y_hat_truncated)
        tmp_loss = loss_fn(y_hat_truncated, y_truncated)
        if weightingMethod == 'mean':
            loss += calc_mean(tmp_loss)
        elif weightingMethod == 'cumulative':
            loss += calc_cumulative(tmp_loss)
        elif weightingMethod == 'only_last':
            loss += calc_only_last(tmp_loss)
        # loss += loss_fn(y_hat_truncated, y_truncated)
        accu_last_token += (y_hat_i[-1] * y[i] > 0).float().mean().item()
        accu_all_tokens += torch.mean((y_hat_truncated * y_truncated > 0).to(torch.float)).item()
    loss /= bs
    accu_all_tokens /= bs
    accu_last_token /= bs
    return {'loss':loss, 'accu_all_tokens':accu_all_tokens, 'accu_last_token':accu_last_token}

def calc_ce_loss(y_hat, y, loss_fn, inds, weightingMethod: Literal['mean', 'cumulative', 'only_last']='mean'):
    '''
    this function calculate the pairwise loss given the predicted reward score and the true choice
    y_hat: (bs, max_token_length, 2), NOTICE: when we use logistic loss, y_hat represents the reward score for the correct response
    y: (bs,)
    inds: [divergence_inds, end_inds]
    '''
    bs = y_hat.size(0)
    loss = 0.
    accu_all_tokens = 0
    accu_last_token = 0
    for i in range(bs):
        # logger.critical(f"inds: {inds}")
        divergence_ind, end_ind = inds[0][i], inds[1][i]
        y_hat_i = y_hat[i] # (max_token_length, 2)
        y_hat_truncated = y_hat_i[divergence_ind:end_ind, :]
        y_truncated = y[i] * torch.ones_like(y_hat_truncated[:, 0])
        y_truncated = y_truncated.long()
        # loss += loss_fn(y_hat_truncated, y_truncated)
        # logger.critical(f"y_hat_truncated: {y_hat_truncated.shape}, y_truncated: {y_truncated.shape}")
        tmp_loss = loss_fn(y_hat_truncated, y_truncated)
        # logger.critical(f"tmp_loss: {tmp_loss}")
        if weightingMethod == 'mean':
            loss += calc_mean(tmp_loss)
        elif weightingMethod == 'cumulative':
            loss += calc_cumulative(tmp_loss)
        elif weightingMethod == 'only_last':
            loss += calc_only_last(tmp_loss)
        accu_last_token += (y_hat_i[-1,:].argmax() == y[i]).float().mean().item()
        accu_all_tokens += (y_hat_truncated.argmax(dim=1) == y_truncated).float().mean().item()
    loss /= bs
    accu_all_tokens /= bs
    accu_last_token /= bs
    return {'loss':loss, 'accu_all_tokens':accu_all_tokens, 'accu_last_token':accu_last_token}

