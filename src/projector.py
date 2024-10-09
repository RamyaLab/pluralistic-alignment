from typing import Sequence
import torch.nn as nn
import torch


class Projector(nn.Module):

    in_dims: int
    out_dims: int
    latent_dims: Sequence[int]
    bias: bool
    dropout_p: float
    activation: str
    identity_map: bool
    use_batchnorm: bool

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        latent_dims: Sequence[int] = tuple([]),
        bias: bool = True,
        dropout_p: float = 0.2,
        activation:str='relu',
        identity_map=False,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.bias = bias
        self.dropout_p = dropout_p
        self.latent_dims = latent_dims
        self.act = None
        self.identity_map = identity_map
        self.use_batchnorm = use_batchnorm
        
        if activation == 'relu':
            self.act = nn.ReLU
        elif activation == 'gelu':
            self.act = nn.GELU
        elif activation == 'linear':
            self.act = nn.Identity
        else:
            raise ValueError(f'no such activation {activation}')
        
        if identity_map == True:
            self.identity = nn.Identity()
            # self.alpha = nn.Parameter(torch.tensor(0.5))

        layer_dims = [in_dims] + list(latent_dims)
        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=self.bias))
            if self.use_batchnorm:  # Add batch normalization layer if enabled
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            layers.extend([
                nn.Dropout(p=self.dropout_p),
                self.act()
            ])
        
        layers.append(nn.Linear(layer_dims[-1], out_dims, bias=self.bias))
        self.layers = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the projector model.

        Args:
            x: The input features.

        Returns:
            torch.Tensor: The projected features.

        """
        if self.identity_map:
            x = self.identity(x) + self.layers(x)
        else:
            x = self.layers(x)
        return x
