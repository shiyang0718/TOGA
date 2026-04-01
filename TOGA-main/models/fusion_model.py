import torch
from torch import nn
import torch.nn.utils as utils

class ConcatFusion(nn.Module):
    def __init__(self, dim_t, dim_a, dim_v, out_c) -> None:

        super().__init__()

        total_dim = dim_t + dim_a + dim_v

        self.fxy = utils.spectral_norm(nn.Linear(total_dim, out_c))

    def forward(self, t, a, v):
        combined = torch.cat([t, a, v], dim=1)
        out = self.fxy(combined)
        return t, a, v, out

class SumFusion(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

    def forward(self, *args): pass


class GatedFusion(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

    def forward(self, *args): pass


class LMF(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()

    def forward(self, *args): pass
