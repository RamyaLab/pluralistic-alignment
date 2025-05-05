import torch
import torch.nn as nn
import torch.nn.functional as F
from ..connector import Connector
from ..projector import Projector
from ..tensor_merger import TensorMerger
import numpy as np

from typing import Literal, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

class ItemLearner(nn.Module):

    connector_x: nn.Module
    projector: Projector
    learner_type: str
    
    def __init__(
        self,
        connector_x: nn.Module,
        projector: Projector,
        learner_type: Literal['dist','dist_normalization','angle','norm','dist_logistic', 'angle_hinge'],
    ) -> None:
        super().__init__()
        self.connector_x = connector_x
        self.learner_type = learner_type
        if learner_type in ['dist','dist_normalization','angle','dist_logistic','angle_hinge']:   # |f(x)-f(u)|_2 or <f(x), f(u)>
            self.projector = projector
        elif learner_type == 'norm':    # # |f(x-u)|_2
            pass
        else:
            raise ValueError(f"Unknown learner_type={learner_type}.")

    def forward(self, x: Tuple[torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        x_left, x_right = x
        x_left = self.connector_x(x_left)
        x_right = self.connector_x(x_right)
        if self.learner_type in ['dist','dist_normalization','angle','dist_logistic','angle_hinge']:  # |f(x)-f(u)|_2 or <f(x), f(u)>
            return self.projector(x_left), self.projector(x_right)
        elif self.learner_type == 'norm':   # # |f(x-u)|_2 (do the self.projector() part in the PreferenceLearner)
            return x_left, x_right
        else:
            raise ValueError(f"Unknown learner_type={self.learner_type}.")
