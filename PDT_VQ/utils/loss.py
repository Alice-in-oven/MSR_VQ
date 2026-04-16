import torch
import torch.nn as nn
import torch.nn.functional as F


def triplet_loss(x, pos, neg, delta=0.):
    dist_pos = F.pairwise_distance(x, pos)
    dist_neg = F.pairwise_distance(x, neg)
    per_point_loss = F.relu(delta + dist_pos - dist_neg)
    return per_point_loss.mean()

def distance_loss(x1, x2, batch_size=1024):
    dist1 = torch.cdist(x1, x1, 2)
    dist2 = torch.cdist(x2, x2, 2)
    dist_loss = torch.sqrt(torch.pow(dist1 - dist2, 2).clamp(min=1e-10))
    return dist_loss.mean(-1)
    
def pairwise_NNs_inner(x):
    dots = torch.matmul(x, x.transpose(-1, -2)) # [M, K, K]
    m, k, _ = x.shape
    dots.view(m, -1)[:, ::(k+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.topk(dots, 1, -1)
    # I: [M, K, 1]
    return I

def uniform_loss(x):
    # x: [M, K, d]
    I = pairwise_NNs_inner(x)
    max_x = torch.gather(x, 1, I.repeat(1, 1, x.shape[-1]))
    distances = F.pairwise_distance(x, max_x)
    loss_uniform = - torch.log(x.shape[-1] * distances).mean()
    return loss_uniform
    # distances = F.pairwise_distance(x, x[I])
    # loss_uniform = - torch.log(x.shape[-1] * distances).mean()
    # return loss_uniform