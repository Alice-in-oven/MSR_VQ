import torch
import torch.nn as nn
from .quantizer import BaseQuantizer
from einops import rearrange
import faiss
import math
from faiss.contrib.inspect_tools import get_pq_centroids


class DPQ(nn.Module):
    def __init__(self, d, M, K, init='uniform', **kwargs):
        super().__init__()
        self.M, self.K = M, K
        self.init = init
        assert d % M == 0
        self.quantizer = BaseQuantizer(d, M, K)

    def init_codebook(self, x, resume=None):
        if self.init == 'uniform':
            print('initialize with uniform')
            nn.init.kaiming_uniform_(self.quantizer.codebook)
        elif self.init == 'faiss':
            nbit = int(math.log2(self.K))
            print(f'initialize with faiss PQ{self.M}x{nbit}')
            pq = faiss.ProductQuantizer(x.shape[-1], self.M, nbit)
            pq.train(x.data.cpu().numpy())
            centroids = torch.as_tensor(get_pq_centroids(pq)).to(x.device)
            self.quantizer.codebook.copy_(centroids)
        else:
            raise NotImplementedError

    def get_codes(self, x):
        return self.quantizer.get_codes(x)
    
    def reconstruction(self, codes):
        x_recon = [self.quantizer.codebook[m, codes[:, m]] for m in range(self.M)]
        x_recon = torch.stack(x_recon)
        x_recon = rearrange(x_recon, 'm n d -> n (m d)')
        return x_recon
    
    def forward(self, x):
        x_recon, codes = self.quantizer(x)
        side_output = rearrange(x_recon, 'n (m d) -> m n d', m=self.M)
        return x_recon, codes, side_output

