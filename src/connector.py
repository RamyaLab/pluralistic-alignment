from .projector import Projector
import torch.nn as nn
import re


# class Connector(nn.Module):
#     def __init__(self, cnct_arch:str, in_dims:int, out_dims:int):
#         super().__init__()
#         # projector_type structure ["mlp?-relu-dropout?-residual","identity"]
#         self.cnct_arch = cnct_arch
        
#         if cnct_arch == 'identity':
#             self.m = nn.Identity()
            
#         pattern = r"mlp(\d+)-(relu|gelu|linear)-dropout(\d+)?(-residual-batchnorm|-batchnorm-residual|-residual|-batchnorm|-nobias)?"
#         match = re.match(pattern, cnct_arch)
        
#         if match:
#             layers = int(match.group(1))
#             act = match.group(2)
#             dropout_p = int(match.group(3))
#             num_digit = len(match.group(3))
#             dropout_p = dropout_p / 10**num_digit
#             # print("match.group(4): ", match.group(4))
#             nobias = False
#             if match.group(4) != None:
#                 residual = True if ("-residual" in match.group(4)) else False
#                 batchnorm = True if ("-batchnorm" in match.group(4)) else False
#                 nobias = True if ("-nobias" in match.group(4)) else False
#             else:
#                 residual = False
#                 batchnorm = False
#             latent_dims = [out_dims] * layers
#             self.m = Projector(
#                 in_dims=in_dims,
#                 out_dims=out_dims,
#                 latent_dims=latent_dims,
#                 bias=not nobias,
#                 dropout_p=dropout_p,
#                 activation=act,
#                 identity_map=residual,
#                 use_batchnorm=batchnorm,
#             )
        
#     def forward(self,x):
#         return self.m(x)


class Connector(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, cnct_arch:str):
        super().__init__()
        pattern = r"mlp(\d+)-(relu|gelu|linear)-dropout(\d+)?(-residual-batchnorm|-batchnorm-residual|-residual|-batchnorm|-nobias)?"
        match = re.match(pattern, cnct_arch)
        if match:
            layers = int(match.group(1))
            act = match.group(2)
            dropout_p = int(match.group(3))
            num_digit = len(match.group(3))
            dropout_p = dropout_p / 10**num_digit
            if match.group(4) != None:
                residual = True if ("-residual" in match.group(4)) else False
                batchnorm = True if ("-batchnorm" in match.group(4)) else False
                nobias = True if ("-nobias" in match.group(4)) else False
            else:
                residual = False
                batchnorm = False
                nobias = False
            latent_dims = [out_dims] * layers
            self.mlp = Projector(
                in_dims=in_dims,
                out_dims=out_dims,
                latent_dims=latent_dims,
                bias=not nobias,
                dropout_p=dropout_p,
                activation=act,
                identity_map=residual,
                use_batchnorm=batchnorm,
            )
        elif cnct_arch == 'identity':
            self.mlp = nn.Identity()
        else:
            raise ValueError(f'no such connection architecture {cnct_arch}')
        
    def __call__(self, x):
        ret = self.mlp(x)
        return ret

if __name__ == "__main__":
    m = Connector(cnct_arch='identity',in_dims=4096,out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp1-relu-dropout2-residual',in_dims=4096,out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp1-relu-dropout2-batchnorm',in_dims=4096,out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp1-relu-dropout2-residual-batchnorm',in_dims=4096,out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp3-gelu-dropout2',in_dims=4096,out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp16-relu-dropout75',in_dims=4096,out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp0-linear-dropout0', in_dims=4096, out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp0-linear-dropout0-nobias', in_dims=4096, out_dims=768)
    print(m)
    m = Connector(cnct_arch='mlp2-linear-dropout0-nobias', in_dims=4096, out_dims=768)
    print(m)

    m = Connector(cnct_arch='mlp2-gelu-dropout0', in_dims=512, out_dims=512)
    count = 0
    for p in m.parameters():
        count += p.numel()
    print(count)