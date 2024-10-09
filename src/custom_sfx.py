import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Optional

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomSoftMax(nn.Module):
    def __init__(
        self, 
        sfx_type: Literal['gumbel_softmax', 'softmax'], 
        temperature: float,
        is_temperature_learnable: bool,
        is_gumbel_hard: Optional[bool]=None, # [True/False] 
        *args, 
        **kwargs,
    ) -> None:

        super().__init__()
        self.sfx_type = sfx_type
        assert not is_temperature_learnable, 'is_temperature_learnable is prohibited in this version, will go to negative'
        self.temperature = nn.Parameter(torch.tensor([float(temperature)]),requires_grad=is_temperature_learnable)
        self.is_gumbel_hard = is_gumbel_hard
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x):   
        # x: (bs, dims)
        if self.sfx_type == 'gumbel_softmax':
            if self.is_gumbel_hard is not None:
                return F.gumbel_softmax(x, tau=self.temperature, hard=self.is_gumbel_hard, dim=1)
            else:
                raise ValueError('is_gumbel_hard is not passed')
        elif self.sfx_type == 'softmax':
            return F.softmax(x/self.temperature, dim=1)
        else:
            raise NotImplementedError(f'{self.sfx_type} is not implemented yet')
        
if __name__ == "__main__":
    
    sfx = CustomSoftMax(sfx_type='gumbel_softmax', temperature=1, is_temperature_learnable=False, is_gumbel_hard=True)
    x = torch.randn(10,3)   # (bs, dims)
    print(x.shape)
    print(sfx(x))
    
    sfx = CustomSoftMax(sfx_type='gumbel_softmax', temperature=1, is_temperature_learnable=True, is_gumbel_hard=True)
    x = torch.randn(10,3)   # (bs, dims)
    print(x.shape)
    print(sfx(x))

    sfx = CustomSoftMax(sfx_type='softmax', temperature=1, is_temperature_learnable=False)
    x = torch.randn(10,3)
    print(sfx(x))
    
    sfx = CustomSoftMax(sfx_type='softmax',temperature=0.01, is_temperature_learnable=True, is_gumbel_hard=None)
    x = torch.randn(10,3)
    print(sfx(x))
    