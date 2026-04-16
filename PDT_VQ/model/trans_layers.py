import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import transpose
from torch.nn.utils.parametrizations import orthogonal


class DefaultTransLayer(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self.d = d
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))


class OrthogonalTrans(DefaultTransLayer):
    def __init__(self, d) -> None:
        super().__init__(d)
        linear = nn.Linear(d, d, bias=False)
        linear2 = nn.Linear(d, d, bias=False)
        orth_linear = orthogonal(linear)
        trans_linear = transpose(linear, linear2)

        self.encoder = orth_linear
        self.decoder = trans_linear


class MLPTrans(DefaultTransLayer):
    def __init__(self, d, d_hidden, steps=1) -> None:
        super().__init__(d)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        for i in range(steps):
            d_in = d if i == 0 else d_hidden
            d_out = d if i == steps - 1 else d_hidden
            self.encoder.extend(nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()))
            self.decoder.extend(nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()))


class NonLinear(nn.Module):
    def __init__(self, d, d_hidden, heads=1, step_norm=True, head_norm=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.non_linear = nn.ModuleList()
        self.step_norm = step_norm
        self.head_norm = head_norm
        for _ in range(heads):
            basic_transform = nn.Sequential(
                    nn.Linear(d, d_hidden),
                    nn.LayerNorm(d_hidden),
                    nn.GELU(),
                    nn.Linear(d_hidden, d)
                )
            if self.step_norm:
                basic_transform.append(Normalize())

            self.non_linear.append(basic_transform) 
        self.to_score = nn.Linear(d, 1)
    
    def forward(self, x):
        xs = []
        for layer in self.non_linear:
            xs.append(layer(x))
        xs = torch.stack(xs, dim=1)
        if self.head_norm:
            score = self.to_score(xs).softmax(1)   # [n, h, 1]
            xs = (xs * score).sum(dim=1)
        else:
            xs = xs.sum(dim=1)
        return xs

class MultiStepNonLinear(nn.Module):
    def __init__(self, d, d_hidden, heads=1, steps=1, step_norm=True, head_norm=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.transform = nn.ModuleList()
        self.steps = steps
        for _ in range(steps):
            self.transform.append(
                NonLinear(d, d_hidden, heads, step_norm, head_norm)
            )
    def forward(self, x, out_step=None, mode='encoder'):
        xs = []
        if mode == 'encoder':
            for i in range(out_step):
                x = x + self.transform[i](x)
                xs.append(x)
        elif mode == 'decoder':
            for i in range(out_step):
                x = x + self.transform[i](x)
                xs.append(x)
        return xs


class MultiStepDistributionTrans(DefaultTransLayer):
    def __init__(self, d, d_hidden, M, steps=1, heads=1, step_norm=True, head_norm=True) -> None:
        super().__init__(d)
        self.steps = steps
        self.heads = heads
        self.M = M  
        self.encoder = MultiStepNonLinear(d, d_hidden, heads, steps, step_norm, head_norm)
        self.decoder = MultiStepNonLinear(d, d_hidden, heads, steps, step_norm, head_norm)

    def encode(self, x, out_step=None):
        if self.training:
            x = x + 0.001 * torch.randn(x.shape).to(x.device)
        if out_step is None:
            out_step = self.steps
        return self.encoder(x, out_step, 'encoder')

    def decode(self, x, out_step=None):
        if out_step is None:
            out_step = self.steps
        return self.decoder(x, out_step, 'decoder')

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        return x
    