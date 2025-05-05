import torch
import torch.nn as nn
import torch.nn.functional as F
from ..connector import Connector
from ..projector import Projector
from ..tensor_initializer import TensorInitializer
from ..tensor_merger import TensorMerger
from ..custom_sfx import CustomSoftMax
import numpy as np
from .itemLearner import ItemLearner
from .userLearner import UserLearner as UserPromptLearner

from collections import defaultdict
from typing import Literal, Optional, Tuple

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasePrefLearner(nn.Module):

    def __init__(
        self,
        d_p: int,   # the latent dims of prompt
        d_x: int,   # the latent dims of items
        d: int,     # the dims of the preference embedding space
        k: int,     # the number of groups
        pref_learner_type: Literal["dist","dist_normalization","angle","norm","dist_logistic","angle_hinge"],
        cnct_arch: str,
        proj_arch: str,
        sfx_type: Literal["gumbel_softmax", "softmax"],
        sfx_temperature: float,
        is_temperature_learnable: bool,
        is_gumbel_hard: Optional[bool]=None,
        is_partition: bool=False,
        seed: int=42,
        **kwargs,
    ):
        super().__init__()
        self.pref_learner_type = pref_learner_type
        self.is_temperature_learnable = is_temperature_learnable
        # initialize all modules
        self.cnct_arch, self.proj_arch = cnct_arch, proj_arch
        self.projector_f = Connector(cnct_arch=proj_arch,in_dims=d_x,out_dims=d)
        self.projectors_gk = [Connector(cnct_arch=proj_arch,in_dims=d_p,out_dims=d) for _ in range(k)]
        connector_p = Connector(cnct_arch=cnct_arch,in_dims=d_p,out_dims=d_p)
        connector_x = Connector(cnct_arch=cnct_arch,in_dims=d_x,out_dims=d_x)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softmax_w = CustomSoftMax(sfx_type=sfx_type, 
                                       temperature=sfx_temperature,
                                       is_temperature_learnable=is_temperature_learnable,
                                       is_gumbel_hard=is_gumbel_hard)
        self.is_partition = is_partition
        # initalize learners with the module components above
        self.item_learner = ItemLearner(
            connector_x=connector_x,
            projector=self.projector_f,
            learner_type=pref_learner_type,
        )
        self.user_learner = UserPromptLearner(
            k=k,
            connector_p=connector_p,
            softmax=self.softmax_w,
            projectors=self.projectors_gk,
            is_partition=self.is_partition,
        )

    def update_trainable_params(self, fix_modules: Tuple[str,...]=()):
        # capture params
        self.trainable_params = defaultdict(list)
        
        if "item_learner_projector" not in fix_modules:
            self.trainable_params["projector"].extend(list(self.item_learner.projector.parameters()))
        
        if "user_prompt_learner_projector" not in fix_modules:
            self.trainable_params["projector"].extend(list(self.user_learner.projectors.parameters()))
        
        if "W" not in fix_modules:
            self.trainable_params["W"] = self.user_learner.W.parameters()

        if self.cnct_arch != "identity":
            self.trainable_params["connector_p"] = self.item_learner.connector_p.parameters()
            self.trainable_params["connector_x"] = self.item_learner.connector_x.parameters()

        if self.pref_learner_type in ["angle","dist_logistic"] and "logit_scale" not in fix_modules:
            self.trainable_params["logit_scale"] = self.logit_scale

        if self.is_temperature_learnable and "temperature" not in fix_modules:
            self.trainable_params["temperature"] = self.softmax_w.temperature

        logger.critical(f"👾 Aha, these are Trainable parameters: {self.trainable_params.keys()}")

    def map_to_pref_embedding_space(self, x):
        u_ids, p, (x_left, x_right) = x
        x_left_prime, x_right_prime = self.item_learner((x_left, x_right))   # (bs ,dims)
        u_prime = self.user_learner(u_ids, p)  # (bs ,dims)
        return x_left_prime, x_right_prime, u_prime

class PrefLearner_dist(BasePrefLearner):    # |f(x)-f(u)|_2
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_to_u_left = torch.linalg.norm(x_left_prime - u_prime, dim=1)
        x_to_u_right = torch.linalg.norm(x_right_prime - u_prime, dim=1)
        return x_to_u_left - x_to_u_right

class PrefLearner_distNorm(BasePrefLearner):    # ||f(x)|-|f(u)||_2
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_left_prime, x_right_prime = x_left_prime/torch.norm(x_left_prime, dim=1, keepdim=True), x_right_prime/torch.norm(x_right_prime, dim=1, keepdim=True)
        u_prime = u_prime/torch.norm(u_prime, dim=1, keepdim=True)
        x_to_u_left = torch.linalg.norm(x_left_prime - u_prime, dim=1)
        x_to_u_right = torch.linalg.norm(x_right_prime - u_prime, dim=1)
        return x_to_u_left - x_to_u_right

class PrefLearner_norm(BasePrefLearner):    # |f(x-u)|_2
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_left_prime_minus_u_prime = (x_left_prime - u_prime)
        x_right_prime_minus_u_prime = (x_right_prime - u_prime)
        ele_1 = self.projector(x_left_prime_minus_u_prime)
        ele_2 = self.projector(x_right_prime_minus_u_prime)
        ele_1 = torch.linalg.norm(ele_1, dim=1)
        ele_2 = torch.linalg.norm(ele_2, dim=1)
        return ele_1 - ele_2

class PrefLearner_angle(BasePrefLearner):   # <f(x),f(u)>
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_left_prime = x_left_prime / torch.norm(x_left_prime, dim=1, keepdim=True)
        x_right_prime = x_right_prime / torch.norm(x_right_prime, dim=1, keepdim=True)
        u_prime = u_prime / torch.norm(u_prime, dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        clamped_logit_scale = torch.clamp(logit_scale, max=100)
        sim_left = torch.diagonal(x_left_prime @ u_prime.T) * clamped_logit_scale   # (bs,)
        sim_right = torch.diagonal(x_right_prime @ u_prime.T) * clamped_logit_scale # (bs,)
        return torch.vstack([sim_left,sim_right]).T   # (bs,2)

class PrefLearner_angle_hinge(BasePrefLearner):   # <f(x),f(u)> - <f(x'),f(u)>
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        x_left_prime, x_right_prime, u_prime = self.map_to_pref_embedding_space(x)
        x_left_prime = x_left_prime / torch.norm(x_left_prime, dim=1, keepdim=True)
        x_right_prime = x_right_prime / torch.norm(x_right_prime, dim=1, keepdim=True)
        u_prime = u_prime / torch.norm(u_prime, dim=1, keepdim=True)
        sim_left = torch.diagonal(x_left_prime @ u_prime.T)     # (bs,)
        sim_right = torch.diagonal(x_right_prime @ u_prime.T)   # (bs,)
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
        x_to_u_left = torch.linalg.norm(x_left_prime - u_prime, dim=1) * clamped_logit_scale
        x_to_u_right = torch.linalg.norm(x_right_prime - u_prime, dim=1) * clamped_logit_scale
        return torch.vstack([x_to_u_left,x_to_u_right]).T   # (bs,2)

def prefLearner_factory(pref_learner_type: Literal["dist","dist_normalization","angle","norm","dist_logistic"]):
    if pref_learner_type == "dist":
        return PrefLearner_dist
    elif pref_learner_type == "dist_normalization":
        return PrefLearner_distNorm
    elif pref_learner_type == "angle":
        return PrefLearner_angle
    elif pref_learner_type == "norm":
        return PrefLearner_norm
    elif pref_learner_type == "dist_logistic":
        return PrefLearner_dist_logistic
    elif pref_learner_type == "angle_hinge":
        return PrefLearner_angle_hinge
    else:
        raise ValueError(f"Unknown user_learner_type: {pref_learner_type}")