#!/usr/bin/env python
# -*-coding:utf-8 -*-

'''
@Desc: This is the implementation of PAL-B
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from ..connector import Connector
from ..tensor_initializer import TensorInitializer
from ..custom_sfx import CustomSoftMax
from .itemLearner import ItemLearner
from .userLearner import UserLearner

from collections import defaultdict
from typing import Literal, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

class BasePrefLearner(nn.Module):
    def __init__(
        self, 
        d_hid: int, 
        d_pref: int, 
        k: int, 
        llm_name: str,
        pref_learner_type: Literal["dist","dist_normalization","angle","norm","dist_logistic","angle_hinge"],
        proj_arch: str,
        initializer_type: Literal["gaussian"],
        is_expectation_norm_init: bool, # the tensor initialization parameters
        sfx_type: Literal["gumbel_softmax", "softmax"],
        sfx_temperature: float,
        is_temperature_learnable: bool,
        is_gumbel_hard: Optional[bool]=None,
        is_partition: bool=False,
        seed: int=42,
        **kwargs
    ):
        super().__init__()
        self.pref_learner_type = pref_learner_type
        self.is_temperature_learnable = is_temperature_learnable
        # init all necessary modules
        model_config = AutoConfig.from_pretrained(llm_name)
        self.llm = AutoModel.from_pretrained(llm_name,from_tf=bool(".ckpt" in llm_name),config=model_config)
        self.tensor_initializer = TensorInitializer(initializer_type, seed, is_expectation_norm_init=is_expectation_norm_init)
        self.projector_f = Connector(cnct_arch=proj_arch,in_dims=d_hid,out_dims=d_pref)
        self.projectors_gk = [Connector(cnct_arch=proj_arch,in_dims=d_hid,out_dims=d_pref) for _ in range(k)]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softmax_w = CustomSoftMax(sfx_type=sfx_type, 
                                       temperature=sfx_temperature,
                                       is_temperature_learnable=is_temperature_learnable,
                                       is_gumbel_hard=is_gumbel_hard)
        self.item_learner = ItemLearner(
            llm = self.llm,
            projector=self.projector_f,
        )
        self.is_partition = is_partition
        self.user_learner = UserLearner(k=k, llm=self.llm, projectors=self.projectors_gk, softmax=self.softmax_w, is_partition=is_partition)
        logger.critical('🛑 Remember to call update_trainable_params() after the model is initialized.')

    def update_trainable_params(self, fix_modules: Tuple[str,...]=()):
        # capture params
        self.trainable_params = defaultdict(list)
        if "llm" not in fix_modules:
            self.trainable_params["llm"] = self.llm.parameters()
        else:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()
        if "itemLearnerProjector" not in fix_modules:
            self.trainable_params["projector_f"].extend(self.item_learner.projector.parameters())
        if "userLearnerProjector" not in fix_modules:
            self.trainable_params["projectors_gk"].extend(list(self.user_learner.projectors.parameters()))
        if "W" not in fix_modules:
            self.trainable_params["W"] = self.user_learner.W.parameters()
        if self.pref_learner_type in ["angle","dist_logistic"] and "logit_scale" not in fix_modules:
            self.trainable_params["logit_scale"] = self.logit_scale
        if self.is_temperature_learnable and "temperature" not in fix_modules:
            self.trainable_params["temperature"] = self.softmax_w.temperature    

    def map_to_pref_embedding_space(self, x):
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
        assert hasattr(self, 'trainable_params'), "💢 Please call update_trainable_params() after the model is initialized."
        uids, prompt, items = x
        x_left_prime, x_right_prime = self.item_learner(items)
        u_prime = self.user_learner(uids, prompt)
        return x_left_prime, x_right_prime, u_prime

class PrefLearner_dist(BasePrefLearner):    # |f(x)-f(u)|_2
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_to_u_left = torch.linalg.norm(x_left_prime - u_prime.unsqueeze(1), dim=-1)
        x_to_u_right = torch.linalg.norm(x_right_prime - u_prime.unsqueeze(1), dim=-1)
        return x_to_u_left - x_to_u_right   # (bs, max_token_length)

class PrefLearner_distNorm(BasePrefLearner):    # ||f(x)|-|f(u)||_2
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_left_prime, x_right_prime = x_left_prime/torch.norm(x_left_prime, dim=1, keepdim=True), x_right_prime/torch.norm(x_right_prime, dim=1, keepdim=True)
        u_prime = u_prime/torch.norm(u_prime, dim=1, keepdim=True)
        x_to_u_left = torch.linalg.norm(x_left_prime - u_prime.unsqueeze(1), dim=-1)
        x_to_u_right = torch.linalg.norm(x_right_prime - u_prime.unsqueeze(1), dim=-1)
        return x_to_u_left - x_to_u_right

class PrefLearner_angle(BasePrefLearner):   # <f(x),f(u)>
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        # logger.critical(f"x_left_prime: {x_left_prime.shape}, x_right_prime: {x_right_prime.shape}, u_prime: {u_prime.shape}")
        x_left_prime = x_left_prime / torch.norm(x_left_prime, dim=-1, keepdim=True)
        x_right_prime = x_right_prime / torch.norm(x_right_prime, dim=-1, keepdim=True)
        u_prime = u_prime / torch.norm(u_prime, dim=-1, keepdim=True)
        # u_prime = u_prime.unsqueeze(1)
        logit_scale = self.logit_scale.exp()
        clamped_logit_scale = torch.clamp(logit_scale, max=100)
        sim_left = (u_prime * x_left_prime).sum(dim=-1) * clamped_logit_scale   # (bs, max_token_length)
        sim_right = (u_prime * x_right_prime).sum(dim=-1) * clamped_logit_scale # (bs, max_token_length)
        # logger.critical(f"sim_left: {sim_left.shape}, sim_right: {sim_right.shape}")
        return torch.stack([sim_left, sim_right], dim=-1)  # # (bs, max_token_length, 2)

class PrefLearner_angle_hinge(BasePrefLearner):   # <f(x),f(u)> - <f(x'),f(u)>
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_left_prime = x_left_prime / torch.norm(x_left_prime, dim=-1, keepdim=True)
        x_right_prime = x_right_prime / torch.norm(x_right_prime, dim=-1, keepdim=True)
        u_prime = u_prime / torch.norm(u_prime, dim=1, keepdim=True)
        u_prime = u_prime.unsqueeze(1)
        logit_scale = self.logit_scale.exp()
        clamped_logit_scale = torch.clamp(logit_scale, max=100)
        sim_left = (u_prime * x_left_prime).sum(dim=-1) * clamped_logit_scale   # (bs,)
        sim_right = (u_prime * x_right_prime).sum(dim=-1) * clamped_logit_scale # (bs,)
        dissim_left = 1 - sim_left
        dissim_right = 1 - sim_right
        return dissim_left-dissim_right

class PrefLearner_dist_logistic(BasePrefLearner):    # |f(x)-f(u)|_2
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        logit_scale = self.logit_scale.exp()
        clamped_logit_scale = torch.clamp(logit_scale, max=100)
        x_to_u_left = torch.linalg.norm(x_left_prime - u_prime.unsqueeze(1), dim=-1) * clamped_logit_scale
        x_to_u_right = torch.linalg.norm(x_right_prime - u_prime.unsqueeze(1), dim=-1) * clamped_logit_scale
        return torch.stack([x_to_u_left, x_to_u_right], dim=-1)  # # (bs, max_token_length, 2)
        
def prefLearner_factory(pref_learner_type: Literal["dist","dist_normalization","angle", "angle_hinge","norm","dist_logistic"]):
    if pref_learner_type == "dist":
        return PrefLearner_dist
    elif pref_learner_type == "dist_normalization":
        return PrefLearner_distNorm
    elif pref_learner_type == "angle":
        return PrefLearner_angle
    elif pref_learner_type == "dist_logistic":
        return PrefLearner_dist_logistic
    elif pref_learner_type == "angle_hinge":
        return PrefLearner_angle_hinge
    else:
        raise ValueError(f"Unknown user_learner_type: {pref_learner_type}")
