import torch
import numpy as np
import torch.nn as nn
from .quantizer import BaseQuantizer
import math
import faiss
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks


class DRQ(nn.Module):
    def __init__(self, d, M, K, init='uniform', **kwargs):
        super().__init__()
        self.M, self.K = M, K
        self.init = init
        self.d = d
        self.d_hidden = d
        self.quantizer = nn.ModuleList([BaseQuantizer(d, 1, K) for _ in range(M)])

    def init_codebook(self, x, resume=None):
        if self.init == 'uniform':
            print('initialize with uniform')
            for q in self.quantizer:
                nn.init.kaiming_uniform_(q.codebook)
        elif self.init == 'faiss':
            nbit = int(math.log2(self.K))
            print(f'initialize with faiss RQ{self.M}x{nbit}')
            rq = faiss.ResidualQuantizer(x.shape[-1], self.M, nbit)
            rq.train(x.data.cpu().numpy())
            centroids = torch.as_tensor(np.array(get_additive_quantizer_codebooks(rq))).to(x.device)
            for i in range(self.M):
                self.quantizer[i].codebook.copy_(centroids[i][None])
        else:
            raise NotImplementedError
    
    def get_codes(self, x):
        return self.encode_decode(x)[1].argmax(dim=-1)
    
    def reconstruction(self, codes):
        x_recon = [q.codebook[:, codes[:, i]] for (i, q) in enumerate(self.quantizer)]
        x_recon = torch.cat(x_recon, dim=0).sum(dim=0)
        return x_recon

    def encode_decode(self, x):
        x_recon, code = self.quantizer[0](x)    # x_recon: [n, d]  code: [n, 1, K]
        side_output = [x_recon]
        codes = [code]
        res = x - x_recon
        for i in range(1, len(self.quantizer)):
            res_recon, code = self.quantizer[i](res)
            codes.append(code)
            x_recon = x_recon + res_recon
            side_output.append(x_recon)
            res = x - x_recon
        return x_recon, torch.cat(codes, dim=1), torch.stack(side_output, 0)  # [n, M, K]

    def forward(self, x):
        x_recon, codes, side_output = self.encode_decode(x)
        return x_recon, codes, side_output