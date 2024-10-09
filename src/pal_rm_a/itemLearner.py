import torch
import torch.nn as nn
import torch.nn.functional as F
from ..connector import Connector
from ..projector import Projector
from ..tensor_merger import TensorMerger
import numpy as np

from typing import Literal

class ItemLearner(nn.Module):
    llm: nn.Module
    projector: nn.Module
    def __init__(self, llm, projector):
        super(ItemLearner, self).__init__()
        self.llm = llm
        self.projector = projector
    def forward(self,x):
        '''
        x = {'left_input_ids': torch.tensor, 'left_attention_mask': torch.tensor,
             'right_input_ids': torch.tensor, 'right_attention_mask': torch.tensor}
        '''
        left_embeds = self.llm(input_ids=x['left_input_ids'], attention_mask=x['left_attention_mask']).last_hidden_state
        right_embeds = self.llm(input_ids=x['right_input_ids'], attention_mask=x['right_attention_mask']).last_hidden_state
        # left embeds shape: (bs, seq_len, hidden_size)
        # right embeds shape: (bs, seq_len, hidden_size)
        shape = left_embeds.shape
        x_left = left_embeds.view(-1, shape[-1])  # (bs*seq_len, hidden_size)
        x_right = right_embeds.view(-1, shape[-1])   # (bs*seq_len, hidden_size)
        projected_x_left, projected_x_right = self.projector(x_left), self.projector(x_right)
        # transform it back to (batch_size, token_length, dim)
        projected_x_left = projected_x_left.view(shape[0], shape[1], -1)
        projected_x_right = projected_x_right.view(shape[0], shape[1], -1)
        return projected_x_left, projected_x_right
