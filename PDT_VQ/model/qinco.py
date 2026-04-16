import torch
from torch import nn
import math
import faiss
from faiss.contrib.inspect_tools import get_additive_quantizer_codebooks

def pairwise_distances(a, b):
    """
    a (torch.Tensor): Shape [na, d]
    b (torch.Tensor): Shape [nb, d]

    Returns (torch.Tensor): Shape [na,nb]
    """
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    return anorms[:, None] + bnorms - 2 * a @ b.T


def compute_batch_distances(a, b):
    """
    a (torch.Tensor): Shape [n, a, d]
    b (torch.Tensor): Shape [n, b, d]

    Returns (torch.Tensor): Shape [n,a,b]
    """
    anorms = (a**2).sum(-1)
    bnorms = (b**2).sum(-1)
    # return anorms.unsqueeze(-1) + bnorms.unsqueeze(1) - 2 * torch.einsum('nad,nbd->nab',a,b)
    return (
        anorms.unsqueeze(-1) + bnorms.unsqueeze(1) - 2 * torch.bmm(a, b.transpose(2, 1))
    )


def assign_batch_multiple(x, zqs):
    """
    Assigns a batch of vectors to a batch of codebooks

    x (torch.Tensor) Shape: [bs x d]
    zqs (torch.Tensor) All possible next quantization vectors per elements in batch. Shape: [bs x ksq x d]

    Returns:
    codes (torch.int64) Indices of selected quantization vector per batch element. Shape: [bs]
    quantized (torch.Tensor) The selected quantization vector per batch element. Shape: [bs x d]
    """
    bs, d = x.shape
    bs, K, d = zqs.shape

    L2distances = compute_batch_distances(x.unsqueeze(1), zqs).squeeze(1)  # [bs x ksq]
    idx = torch.argmin(L2distances, dim=1).unsqueeze(1)  # [bsx1]
    quantized = torch.gather(zqs, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, d))
    return idx.squeeze(1), quantized.squeeze(1)


def assign_to_codebook(x, c, bs=16384):
    """find the nearest centroid in matrix c for all the vectors
    in matrix x. Compute by batches if necessary to spare GPU memory
    (bs is the batch size)"""
    nq, d = x.shape
    nb, d2 = c.shape
    assert d == d2
    if nq * nb < bs * bs:
        # small enough to represent the whole distance table
        dis = pairwise_distances(x, c)
        return dis.argmin(1)

    # otherwise tile computation to avoid OOM
    res = torch.empty((nq,), dtype=torch.int64, device=x.device)
    cnorms = (c**2).sum(1)
    for i in range(0, nq, bs):
        xnorms = (x[i : i + bs] ** 2).sum(1, keepdim=True)
        for j in range(0, nb, bs):
            dis = xnorms + cnorms[j : j + bs] - 2 * x[i : i + bs] @ c[j : j + bs].T
            dmini, imini = dis.min(1)
            if j == 0:
                dmin = dmini
                imin = imini
            else:
                (mask,) = torch.where(dmini < dmin)
                dmin[mask] = dmini[mask]
                imin[mask] = imini[mask] + j
        res[i : i + bs] = imin
    return res

####################################################################
# The base QINCo model
#####################################################################


class QINCoStep(nn.Module):
    """
    One quantization step for QINCo.
    Contains the codebook, concatenation block, and residual blocks
    """

    def __init__(self, d, K, L, h):
        nn.Module.__init__(self)

        self.d, self.K, self.L, self.h = d, K, L, h

        self.codebook = nn.Embedding(K, d)
        self.MLPconcat = nn.Linear(2 * d, d)

        self.residual_blocks = []
        for l in range(L):
            residual_block = nn.Sequential(
                nn.Linear(d, h, bias=False), 
                nn.ReLU(),
                nn.Linear(h, d, bias=False)    ###
            )
            self.add_module(f"residual_block{l}", residual_block)
            self.residual_blocks.append(residual_block)

    def reset_identity(self):
        nn.init.zeros_(self.MLPconcat.weight)
        if self.MLPconcat.bias is not None:
            nn.init.zeros_(self.MLPconcat.bias)
        for residual_block in self.residual_blocks:
            for layer in residual_block:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def decode(self, xhat, codes):
        zqs = self.codebook(codes)
        cc = torch.concatenate((zqs, xhat), 1)
        zqs = zqs + self.MLPconcat(cc)

        for residual_block in self.residual_blocks:
            zqs = zqs + residual_block(zqs)

        return zqs

    def encode(self, xhat, x):
        # we are trying out the whole codebook
        zqs = self.codebook.weight
        K, d = zqs.shape
        bs, d = xhat.shape

        # repeat so that they are of size bs * K
        zqs_r = zqs.repeat(bs, 1, 1).reshape(bs * K, d)
        xhat_r = xhat.reshape(bs, 1, d).repeat(1, K, 1).reshape(bs * K, d)

        # pass on batch of size bs * K
        cc = torch.concatenate((zqs_r, xhat_r), 1)
        zqs_r = zqs_r + self.MLPconcat(cc)

        for residual_block in self.residual_blocks:
            zqs_r = zqs_r + residual_block(zqs_r)

        # possible next steps
        zqs_r = zqs_r.reshape(bs, K, d) + xhat.reshape(bs, 1, d)
        codes, xhat_next = assign_batch_multiple(x, zqs_r)

        return codes, xhat_next - xhat


class QINCo(nn.Module):
    """
    QINCo quantizer, built from a chain of residual quantization steps
    """

    def __init__(self, d, M, K, h=256, L=1, init='uniform', identity_init=False, **kwargs):
        nn.Module.__init__(self)
        self.d, self.K, self.L, self.M, self.h = d, K, L, M, h
        self.init = init
        self.identity_init = identity_init

        self.codebook0 = nn.Embedding(K, d)

        self.steps = []
        for m in range(1, M):
            step = QINCoStep(d, K, L, h)
            self.add_module(f"step{m}", step)
            self.steps.append(step)
        if self.identity_init:
            for step in self.steps:
                step.reset_identity()

    def init_codebook(self, x, resume=None):
        if self.init == 'uniform':
            print('initialize with uniform')
            nn.init.kaiming_uniform_(self.codebook0.weight)
            for step in self.steps:
                nn.init.kaiming_uniform_(step.codebook.weight)
        elif self.init == 'faiss':
            nbit = int(math.log2(self.K))
            print(f'initialize with faiss RQ{self.M}x{nbit}')
            rq = faiss.ResidualQuantizer(x.shape[-1], self.M, nbit)
            rq.train(x.data.cpu().numpy())
            centroids = torch.as_tensor(get_additive_quantizer_codebooks(rq)).to(x.device)
            self.codebook0.weight.copy_(centroids[0])
            for i in range(1, self.M):
                self.steps[i-1].codebook.weight.copy_(centroids[i])
        elif self.init == 'resume':
            if resume is not None:
                print(f'initialize with pretrained codebook')
                state_dict = torch.load(resume)
                for k in state_dict.keys():
                    if 'codebook' in k:
                        m = int(k.split('.')[-2])
                        if m == 0:
                            print('init codenbook 0')
                            self.codebook0.weight.copy_(state_dict[k].to(x.device)[0])
                        else:
                            print(f'init codenbook {m}')
                            self.steps[m-1].codebook.weight.copy_(state_dict[k].to(x.device)[0])
            else:
                assert f'no checkpoint in {resume}'
    
    def get_codes(self, x):
        """
        Encode a batch of vectors x to codes of length M.
        If this function is called from IVF-QINCo, codes are 1 index longer,
        due to the first index being the IVF index, and codebook0 is the IVF codebook.
        """
        M = len(self.steps) + 1
        bs, d = x.shape
        codes = torch.zeros(bs, M, dtype=int, device=x.device)

        # if code0 is None:
        #     # at IVF training time, the code0 is fixed (and precomputed)
        code0 = assign_to_codebook(x, self.codebook0.weight)

        codes[:, 0] = code0
        xhat = self.codebook0.weight[code0]

        for i, step in enumerate(self.steps):
            codes[:, i + 1], toadd = step.encode(xhat, x)
            xhat = xhat + toadd

        return codes

    def reconstruction(self, codes):
        xhat = self.codebook0(codes[:, 0])
        for i, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, i + 1])
        return xhat
    
    def decode(self, codes):
        xhat = self.codebook0(codes[:, 0])
        for i, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, i + 1])
        return xhat

    def encode(self, x, code0=None):
        """
        Encode a batch of vectors x to codes of length M.
        If this function is called from IVF-QINCo, codes are 1 index longer,
        due to the first index being the IVF index, and codebook0 is the IVF codebook.
        """
        M = len(self.steps) + 1
        bs, d = x.shape
        codes = torch.zeros(bs, M, dtype=int, device=x.device)

        if code0 is None:
            # at IVF training time, the code0 is fixed (and precomputed)
            code0 = assign_to_codebook(x, self.codebook0.weight)

        codes[:, 0] = code0
        xhat = self.codebook0.weight[code0]

        for i, step in enumerate(self.steps):
            codes[:, i + 1], toadd = step.encode(xhat, x)
            xhat = xhat + toadd

        return codes, xhat

    def forward(self, x):
        with torch.no_grad():
            codes, xhat = self.encode(x)
        # then decode step by step and collect losses
        xhat = self.codebook0(codes[:, 0])
        side_output = [xhat]
        
        for i, step in enumerate(self.steps):
            xhat = xhat + step.decode(xhat, codes[:, i + 1])
            side_output.append(xhat)
        
        # return codes, xhat, losses
        return xhat, codes, side_output
