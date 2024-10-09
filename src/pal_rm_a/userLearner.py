import torch
import torch.nn as nn
import torch.nn.functional as F
from ..connector import Connector
from ..projector import Projector
from ..tensor_initializer import TensorInitializer
from ..custom_sfx import CustomSoftMax
import numpy as np
import warnings

from typing import Literal

import logging
logger = logging.getLogger(__name__)

class baseUserLearner(nn.Module):
    
    d_q: int    # the dims of the original embedding space
    d: int      # the dims of the preference embedding space
    k: int      # the number of groups
    learner_type: str
    projector: Projector
    u_id_set: set
    softmax: nn.Module

    def __init__(
        self,
        d_q: int,
        d: int,
        k: int,
        learner_type: Literal["Pw", "f(Pw)", "f(P)w", "|f(P)|w"],
        projector: Projector,
        tensor_initializer: TensorInitializer,
        softmax: nn.Module,
    ):
        super().__init__()

        self.d_q = d_q
        self.d = d
        self.k = k
        self.learner_type = learner_type
        self.projector = projector
        self.u_id_set = set()
        self.softmax = softmax
        self.tmp_store_user_ideal_points = None
        # init P and W
        if self.learner_type == "Pw":
            self.P = nn.Parameter(tensor_initializer(dim=self.d, size=self.k), requires_grad=True)
        elif self.learner_type in ["f(Pw)", "f(P)w", "|f(P)|w"]:
            self.P = nn.Parameter(tensor_initializer(dim=self.d_q, size=self.k), requires_grad=True)
        else:
            raise ValueError(f'Unknown learner type for UserLearner: {self.learner_type}')
        self.W = nn.ParameterDict()

    def init_weight(self, u_ids:list, reinit:bool=False):
        for u_id in u_ids:
            if u_id not in self.u_id_set or reinit:
                self.W[u_id] = nn.Parameter(
                    torch.randn((self.k), dtype=torch.float32),
                    requires_grad=True,
                ).to(self.P.device)
                self.u_id_set.add(u_id)
            else:
                logger.warning('👋 wait? same user?')

    def get_sfx_w(self, u_ids:list):
        w = torch.stack([self.W[key] for key in u_ids], dim=0)   # (bs, k)
        w = self.softmax(w)
        return w

    def return_user_ideal_points(self):
        assert self.learner_type != "|f(P)|w", "we don't need to bound this learner_type"
        return self.tmp_store_user_ideal_points

class UserLearner_FPw(baseUserLearner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, u_ids:list):
        w = self.get_sfx_w(u_ids)
        res = self.projector(w @ self.P)
        self.tmp_store_user_ideal_points = res
        return res

class UserLearner_FP_w(baseUserLearner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, u_ids:list):
        w = self.get_sfx_w(u_ids)
        res = w @ self.projector(self.P)
        self.tmp_store_user_ideal_points = res
        return res

class UserLearner_Pw(baseUserLearner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, u_ids:list):
        w = self.get_sfx_w(u_ids)
        res = w @ self.P
        self.tmp_store_user_ideal_points = res
        return res

class UserLearner_AbsoluteFP_w(baseUserLearner):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, u_ids:list):
        w = self.get_sfx_w(u_ids)
        fP = self.projector(self.P)
        fP_norm = fP / torch.norm(fP, dim=1)
        res = w @ fP_norm
        return res

def userLearner_factory(user_learner_type: Literal["Pw", "f(Pw)", "f(P)w", "|f(P)|w"]):
    if user_learner_type == "Pw":
        return UserLearner_Pw
    elif user_learner_type == "f(P)w":
        return UserLearner_FP_w
    elif user_learner_type == "f(Pw)":
        return UserLearner_FPw
    elif user_learner_type == "|f(P)|w":
        return UserLearner_AbsoluteFP_w
    else:
        raise ValueError(f"Unknown user_learner_type: {user_learner_type}")

if __name__ == '__main__':
    projector = Connector('mlp4-relu-dropout5',in_dims=1024, out_dims=1024)
    tensor_initializer = TensorInitializer('gaussian', 42, is_expectation_norm_init=True)
    sfx_fn = CustomSoftMax(
        sfx_type='gumbel_softmax', 
        temperature=1, 
        is_temperature_learnable=False,
        is_gumbel_hard=True,
    )
    userLearnerClass = userLearner_factory('f(P)w')
    user_learner = userLearnerClass(d_q=1024, d=1024, k=2, learner_type='f(Pw)', 
                               projector=projector, tensor_initializer=tensor_initializer,
                               softmax=sfx_fn)
    for i in range(10):
        user_learner.init_weight(str(i))

    print(f"{user_learner.P.shape = }")
    print(f"{next(user_learner.W.parameters()).shape = }")
    print(f"{user_learner.get_sfx_w(['0','1','2','3','4']).shape = }")
    print(f"{user_learner(['0','1','2','3','4']).shape = }")