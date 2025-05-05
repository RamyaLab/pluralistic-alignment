import torch
import torch.nn as nn
import torch.nn.functional as F
from ..connector import Connector
from ..projector import Projector
from ..tensor_initializer import TensorInitializer
from ..custom_sfx import CustomSoftMax
import numpy as np
import warnings

from typing import Literal, Optional
import logging
logger = logging.getLogger(__name__)
import warnings

class UserLearner(nn.Module):
    
    k: int
    connector_p: nn.Module
    projectors: list[Projector]
    u_id_set: set
    softmax: nn.Module
    is_partition: bool

    def __init__(
        self,
        k: int,
        connector_p: nn.Module,
        softmax: nn.Module,
        projectors: list[Projector],
        is_partition: bool=False,
    ):
        super().__init__()

        self.k = k
        self.softmax = softmax
        # init user_id registration table and user weights dictionary
        self.u_id_set = set()
        self.W = nn.ParameterDict()
        self.tmp_store_user_ideal_points = None
        # register all k projectors in the moduledict
        assert len(projectors) == k, f"The num of projectors should match up with num of groups: {k} != {len(projectors)}"
        self.projectors = nn.ModuleDict()
        for i in range(k):
            self.projectors[str(i)] = projectors[i]
        self.connector_p = connector_p
        self.is_partition = is_partition
        if self.is_partition:
            self.is_argmax = False  # if this model is partition version, add the argmax property
        # the effect of connector_p
        # 1. reduce the dim of latent_promtps
        # 2. common base model h for all groups

    def init_weight(self, u_ids:list, reinit:bool=False):
        for u_id in u_ids:
            if u_id not in self.u_id_set or reinit:
                self.W[u_id] = nn.Parameter(
                    torch.randn((self.k), dtype=torch.float32),
                    requires_grad=True,
                )#.to(next(self.projectors[str(0)].parameters()).device)
                self.u_id_set.add(u_id)
            else:
                warnings.warn('👋 wait? same user? Be careful~')
    
    def get_sfx_w(self, u_ids:list):
        w = torch.stack([self.W[key] for key in u_ids], dim=0)   # (bs, k)
        w = self.softmax(w)
        return w

    def get_hardmax_w(self, u_ids:list):
        w = torch.stack([self.W[key] for key in u_ids], dim=0)
        w = F.one_hot(w.argmax(dim=1), num_classes=self.k).float()  # (bs, k)
        return w
    
    def infer_gk(self, latent_prompts):
        # get g_1(prompt), g_2(prompt), ... , g_k(prompt)
        # where g_i(prompt) in the shape of (bs, dims)
        logits = torch.stack([g(self.connector_p(latent_prompts)) for g in self.projectors.values()],dim=1)
        return logits   # (bs, k, dims)
    
    def return_user_ideal_points(self):
        if self.tmp_store_user_ideal_points == None:
            raise ValueError('No user ideal points stored')
        return self.tmp_store_user_ideal_points

    def forward(self, u_ids, latent_prompts):
        prompt_logits = self.infer_gk(latent_prompts)   # (bs, k, dims)
        if self.is_partition and self.is_argmax:
            w = self.get_hardmax_w(u_ids)   # (bs, k)
        else:
            w = self.get_sfx_w(u_ids)       # (bs, k)
        w = w.unsqueeze(1)  # (bs, 1, k)
        y_hat = torch.bmm(w, prompt_logits) # (bs, 1, dims)
        y_hat = y_hat.squeeze(1)    # (bs, dims)
        self.tmp_store_user_ideal_points = y_hat
        return y_hat

    def eval(self):
        super().eval()
        if self.is_partition:
            warnings.warn("🤖 UserPromptLearner(Partition version) is in eval mode: argmax")
            self.is_argmax = True
        else:
            warnings.warn("🤖 UserPromptLearner(Mixture version) is in eval mode: sfx")
    
    def train(self, mode: bool = True):
        
        super().train(mode)
        
        if mode:
            if self.is_partition:
                warnings.warn("🤖 UserPromptLearner(Partition version) is in train mode: sfx")
                self.is_argmax = False
            else:
                warnings.warn("🤖 UserPromptLearner(Mixture version) is in train mode: sfx")
        else:
            if self.is_partition:
                warnings.warn("🤖 UserPromptLearner(Partition version) is in eval mode: argmax")
                self.is_argmax = True
            else:
                warnings.warn("🤖 UserPromptLearner(Mixture version) is in eval mode: sfx")
