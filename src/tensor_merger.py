import torch


class TensorMerger:
    
    def __init__(self, merger_type) -> None:
        self.merger_type = merger_type

    def concat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y], dim=1)

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.merger_type == 'concat':
            return self.concat(x,y)
        else:
            raise ValueError(f'Unknown merger type: {self.merger_type}')