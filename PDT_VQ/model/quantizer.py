import torch
import torch.nn as nn
from .utils import GumbelSoftmax
from einops import rearrange

class BaseQuantizer(nn.Module):
    def __init__(self, d, groups, centroids, **kwargs):
        #print(f"d: {d},groups: {groups}, centroids: {centroids}")
        super().__init__()
        assert d % groups == 0
        dsub = d // groups
        self.groups = groups
        self.centroids = centroids

        self.codebook = nn.Parameter(torch.randn(groups, centroids, dsub), requires_grad=True)
        self.log_temperatures = nn.Parameter(data=torch.zeros(groups), requires_grad=True)
        self.gumbel_softmax = GumbelSoftmax(**kwargs)

    def compute_score(self, x, add_temperatures=True):
        
        x = rearrange(x, 'n (g d) -> n g d', g=self.groups) # [n g d]
        norm_x = torch.sum(x ** 2, dim=-1, keepdim=True)  # [n g 1] ||x||
        norm_c = torch.sum(self.codebook ** 2, dim=-1).unsqueeze(0)  # [1 g k] ||c||
     #(f"x shape:{x.permute(1, 0, 2).shape},codebook shape: {self.codebook.permute(0, 2, 1).shape}")
        dot = torch.matmul(x.permute(1, 0, 2), self.codebook.permute(0, 2, 1))  # [g n d] x [g d k] -> [g n k]
        score = - norm_x + 2 * dot.permute(1, 0, 2) - norm_c  # [n g k]
        if add_temperatures:
            score *= torch.exp(-self.log_temperatures[:, None])
        return score
    
    def get_codes(self, x):
        return self.compute_score(x).argmax(dim=-1) # [n g]
    
    def reconstruction(self, codes):
        x_recon = [self.codebook[g, codes[:, g]] for g in range(self.groups)]
        x_recon = torch.stack(x_recon)
        x_recon = rearrange(x_recon, 'g n d -> n (g d)')
        return x_recon

    def forward(self, x):
        score = self.compute_score(x)
        score = self.gumbel_softmax(score)  # [n g k]
        x_recon = torch.einsum('ngk,gkd->ngd', score, self.codebook)
        x_recon = rearrange(x_recon, 'n g d -> n (g d)')
        return x_recon, score