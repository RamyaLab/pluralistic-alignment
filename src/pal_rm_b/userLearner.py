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

class UserLearner(nn.Module):
    
    k: int      # the number of groups
    llm: nn.Module
    projectors: list[Projector]
    u_id_set: set
    softmax: nn.Module
    is_partition: bool

    def __init__(
        self,
        k: int,
        llm: nn.Module,
        projectors: list[Projector],
        softmax: nn.Module,
        is_partition: bool=False,
    ):
        super().__init__()

        self.k = k
        self.llm = llm
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
        self.is_partition = is_partition

    def init_weight(self, u_ids:list, reinit:bool=False):
        for u_id in u_ids:
            if u_id not in self.u_id_set or reinit:
                self.W[u_id] = nn.Parameter(
                    torch.randn((self.k), dtype=torch.float32),
                    requires_grad=True,
                ).to(next(self.projectors[str(0)].parameters()).device)
                self.u_id_set.add(u_id)
            else:
                logger.warning('👋 wait? same user?')

    def get_sfx_w(self, u_ids:list):
        w = torch.stack([self.W[key] for key in u_ids], dim=0)   # (bs, k)
        w = self.softmax(w)
        return w

    def get_hardmax_w(self, u_ids:list):
        w = torch.stack([self.W[key] for key in u_ids], dim=0)
        w = F.one_hot(w.argmax(dim=1), num_classes=self.k).float()  # (bs, k)
        return w

    def infer_gk(self, prompt_tokens, rm_cached=None):
        '''
        prompt_tokens: {'input_ids': torch.tensor, 'attention_mask': torch.tensor}
        If you want to activate rm_cached, please pass in the rm_cached dict or empty dict.
        '''
        input_ids = prompt_tokens['input_ids']
        attention_mask = prompt_tokens['attention_mask']
        
        if rm_cached is None:
            embeds = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        else:
            res = self.llm(
                input_ids=input_ids[:, -1:],
                # attention_mask=attention_mask,
                past_key_values=rm_cached["user_learner"],
                use_cache=True 
            )
            rm_cached["user_learner"] = res.past_key_values
            embeds = res.last_hidden_state

        # embeds shape: (bs, seq_len, hid_dim)
        shape = embeds.shape
        # TODO: should we only use the last hidden state?
        embeds = embeds.view(-1, shape[-1])  # (bs*seq_len, hid_dim)
        # g(embeds) shape: (bs*seq_len, hid_dim) -> (bs*seq_len, pref_dim)
        logits = torch.stack([g(embeds).view(shape[0], shape[1], -1) for g in self.projectors.values()],dim=1)
        if rm_cached is None:
            return logits
        else:
            return logits, rm_cached   # (bs, k, seq_len, hidden_size)

    def return_user_ideal_points(self):
        if self.tmp_store_user_ideal_points == None:
            raise ValueError('No user ideal points stored')
        return self.tmp_store_user_ideal_points

    def forward(self, u_ids, prompt_tokens):    # only pass the prompt tokens
        '''
        u_ids: list of user ids
        prompt_tokens: {'input_ids': torch.tensor, 'attention_mask': torch.tensor}
        '''
        prompt_logits = self.infer_gk(prompt_tokens)    # (bs, k, seq_len, dims)
        if self.is_partition and self.is_argmax:
            w = self.get_hardmax_w(u_ids)   # (bs, k)
        else:
            w = self.get_sfx_w(u_ids)   # (bs, k)
        w = w.unsqueeze(-1).unsqueeze(-1)   # (bs, k, 1, 1)
        # w @ prompt_logits: (bs, k, seq_len, dims)
        # logger.critical(f'w device: {w.device}, prompt_logits device: {prompt_logits.device}')
        y_hat = (w * prompt_logits).sum(dim=1)
        # y_hat shape: (bs, seq_len, dims)
        self.tmp_store_user_ideal_points = y_hat
        return y_hat
    
    def eval(self):
        super().eval()
        if self.is_partition:
            warnings.warn("🤖 UserPromptLearner(Partition version) is in eval mode: argmax")
            self.is_argmax = True
        else:
            warnings.warn("🤖 UserPromptLearner(Mixture version) is in eval mode: sfx")
            self.is_argmax = False
    
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            if self.is_partition:
                warnings.warn("🤖 UserPromptLearner(Partition version) is in train mode: sfx")
                self.is_argmax = False
            else:
                warnings.warn("🤖 UserPromptLearner(Mixture version) is in train mode: sfx")
                self.is_argmax = False
        else:
            if self.is_partition:
                warnings.warn("🤖 UserPromptLearner(Partition version) is in eval mode: argmax")
                self.is_argmax = True
            else:
                warnings.warn("🤖 UserPromptLearner(Mixture version) is in eval mode: sfx")
                self.is_argmax = False