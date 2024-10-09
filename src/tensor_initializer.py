import numpy as np
import torch

class TensorInitializer:

    def __init__(self, type: str, seed: int, is_expectation_norm_init: bool = False):
        self.initializer_type = type
        self.rng = np.random.default_rng(seed)
        self.is_expectation_norm_init = is_expectation_norm_init

    def gaussian_initializer(
            self,
            dim: int,
            size: int,
        ) -> torch.Tensor:

        mean = np.zeros(dim)
        if self.is_expectation_norm_init:
            # expectation normalization
            cov = 1 / dim * np.eye(dim) 
            return torch.tensor(self.rng.multivariate_normal(mean, cov, size), dtype=torch.float32)#.float()
        else:   
            # enforced normalization
            cov = np.eye(dim)
            unnorm_tensor = torch.tensor(self.rng.multivariate_normal(mean, cov, size), dtype=torch.float32)#.float()
            return unnorm_tensor / torch.norm(unnorm_tensor, dim=1, keepdim=True)

    def __call__(self, *args, **kwargs):
        if self.initializer_type == 'gaussian':
            return self.gaussian_initializer(*args, **kwargs)
        else:
            raise ValueError(f'Unknown initializer type: {self.initializer_type}')
