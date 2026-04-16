import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization


def to_one_hot(y, depth):
    y_flat = y.to(torch.int64).view(-1, 1)
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot

def gumbel_noise(*sizes, eps=1e-20, **kwargs):
    U = torch.rand(*sizes, **kwargs)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(dist, noise=1.0, hard=True, **kwargs):
    if noise != 0:
        z = gumbel_noise(*dist.shape, device=dist.device, dtype=dist.dtype)
        dist = dist + noise * z

    probs_gumbel = torch.softmax(dist, dim=-1)

    if hard:
        _, argmax_indices = torch.max(probs_gumbel, dim=-1)
        hard_argmax_onehot = to_one_hot(argmax_indices, depth=dist.shape[-1])
        probs_gumbel = (hard_argmax_onehot - probs_gumbel).detach() + probs_gumbel

    return probs_gumbel

class GumbelSoftmax(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.opts = kwargs

    def forward(self, dist, **kwargs):
        opts = dict(self.opts)
        if not self.training:
            opts['noise'] = 0.0
            opts['hard'] = True
        return gumbel_softmax(dist, **opts, **kwargs)

def check_numpy(x):
    """ Makes sure x is a numpy array """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
class _Transpose(nn.Module):
    def __init__(self, module1 : nn.Module):
        super().__init__()
        self.module1 = module1

    def forward(self, x):
        weight1 = getattr(self.module1, 'weight', None)
        return weight1.transpose(-1,-2)
    
def transpose(module1: nn.Module, 
              module2: nn.Module, 
              name: str = 'weight', ) -> nn.Module:
    
    register_parametrization(module2, name, _Transpose(module1), unsafe=True)
    return module2