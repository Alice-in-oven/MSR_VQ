import numpy as np
import torch
import torch.nn.functional as F
from .metric import eval_recall
from .get_nn import get_nearestneighbors
from .logger import Logger
from einops import rearrange, repeat
import torch
from typing import Tuple


def train_one_epoch(model, xt, optimizer, args):

    perm = np.random.permutation(xt.shape[0])
    model.train()

    sum_loss = 0
    iters = len(list(range(0, xt.shape[0], args.batch_size)))
    for iter, i0 in enumerate(range(0, xt.shape[0], args.batch_size)):
        i1 = min(i0 + args.batch_size, xt.shape[0])
        data_idx = perm[i0:i1]
        ins = xt[data_idx]
      #  print(f"train one epoch input shape: {ins.shape},data_xt shape:{xt.shape}")
        optimizer.zero_grad()
        loss = model(ins).mean()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
    return sum_loss / iters

@torch.no_grad()
def compute_mse(model, xval, args):
    model.eval()
    all_mse = []
    for iter, i0 in enumerate(range(0, xval.shape[0], args.batch_size)):
        i1 = min(i0 + args.batch_size, xval.shape[0])
        ins = xval[i0:i1]
        all_mse.append(model(ins))
    all_mse = torch.cat(all_mse, 0)
    all_mse = all_mse.mean()
    return all_mse.item()

@torch.no_grad()
def eval(model, xt, xb, xq, groundtruth, args):
    recalls_upbound = inference_exact(model, xt, xb, xq, groundtruth, args)
    recalls = [(-1, recalls_upbound)]
    recalls_rerank = inference_with_reranking(model, xt, xb, xq, groundtruth, args)
    recalls += recalls_rerank
    return recalls
def inference_vetor(model, x, args):
    x_trans = batch_forward(model.encode, x, args.batch_size, args.steps)[-1]
    codes = batch_forward(model.get_codes, x_trans, args.batch_size)
    x_recon_vq = batch_forward(model.reconstruction, codes, args.batch_size)
    x_recon = batch_forward(model.decode, x_recon_vq, args.batch_size)[-1]
    return x_recon
def inference_exact(model, xt, xb, xq, groundtruth, args):
    xb_trans = batch_forward(model.encode, xb, args.batch_size, args.steps)[-1]
    codes = batch_forward(model.get_codes, xb_trans, args.batch_size)
    xb_recon_vq = batch_forward(model.reconstruction, codes, args.batch_size)
    xb_recon = batch_forward(model.decode, xb_recon_vq, args.batch_size)[-1]
    pred_nn = get_nearestneighbors(xb_recon.data.cpu().numpy(), xq.data.cpu().numpy(), 100, args.device)
    recalls = eval_recall(pred_nn, groundtruth, [[1,1], [1,10], [1,100]])
    return recalls

def inference_with_reranking(model, xt, xb, xq, groundtruth, args):
    xb_trans = batch_forward(model.encode, xb, args.batch_size, args.steps)[-1]
    codes = batch_forward(model.get_codes, xb_trans, args.batch_size)
    if args.vq_type == 'qinco':
        xt_trans = batch_forward(model.encode, xt, args.batch_size, args.steps)[-1]
        xt_codes = batch_forward(model.get_codes, xt_trans, args.batch_size)
        fixed_codebook = compute_fixed_codebooks(xt_trans.cpu().numpy(), xt_codes.cpu().numpy(), model.K)
        xb_recon = reconstruct_from_fixed_codebooks(codes.cpu().numpy(), fixed_codebook)
        xb_recon = torch.from_numpy(xb_recon).to(xb.device)
        xb_recon_model = batch_forward(model.reconstruction, codes, args.batch_size)
    else:
        xb_recon = batch_forward(model.reconstruction, codes, args.batch_size)
        xb_recon_model = xb_recon

    xq_trans = batch_forward(model.encode, xq, args.batch_size, args.steps)[-1]

    pred_nn = get_nearestneighbors(xb_recon.data.cpu().numpy(), xq_trans.data.cpu().numpy(), args.L, args.device)
    pred_nn = torch.from_numpy(pred_nn).to(xb_recon.device)  # [nq, L]
    recalls = []
    if args.re_rank:
        for L in 10, 20, 50, 100, 200, 500, 1000:
            one_step_size = max(1, int((args.batch_size / L)))  # reduce the GPU memory usage
            pred_nn_L = re_ranking(xb_recon_model, xq, pred_nn[:, :L], model.decode, L, args.steps, batch_size=args.batch_size, one_step_size=one_step_size)
    
            recalls.append((L, eval_recall(pred_nn_L.data.cpu().numpy(), groundtruth, [[1,1], [1,10], [1,100]])))
    return recalls

def re_ranking(xb, xq, pred_nn, decoder, L, i_step, batch_size=256, one_step_size=100):
    """
    xb: base vector reconstructed by the sub-codebook
    xq: original query vector
    pred_nn: coarsely predicted nn neighbor
    decoder: decoder module
    L: the length of coarsely predicted nn neighbor
    """
    # decoding process
    final_nn = []
    for k, i0 in enumerate(range(0, pred_nn.shape[0], one_step_size)):
        selected_pre_nn = pred_nn[i0:i0 + one_step_size]
        selected_xq = xq[i0:i0 + one_step_size]
        selected_xb = xb[selected_pre_nn]
        selected_xb = rearrange(selected_xb, 'n k d -> (n k) d')
        selected_xb_decode = batch_forward(decoder, selected_xb, batch_size, i_step)[-1]
        selected_xb_decode = rearrange(selected_xb_decode, '(n k) d -> n k d', k=L)
        dist = F.pairwise_distance(selected_xb_decode, selected_xq[:, None], 2)
        _, idx = torch.sort(dist, -1)  # [nq, L]
        final_nn.append(torch.gather(selected_pre_nn, 1, idx)[:, :100]) # for recall 1@100
    final_nn = torch.cat(final_nn, 0)
    return final_nn

def batch_forward(model, x_all, bs, i_step=None):
    multi_out = True
    for k, i0 in enumerate(range(0, x_all.shape[0], bs)):
        x = x_all[i0:i0 + bs]
        if i_step is None:
            output = model(x)
        else:
            output = model(x, i_step)
        if not isinstance(output, Tuple) and not isinstance(output, list):
            multi_out = False
            output = [output]
        if k == 0:
            results = [[output[i]] for i in range(len(output))]
        else:
            for i in range(len(output)):
                results[i].append(output[i])
    return [torch.cat(results[i]) for i in range(len(results))] if multi_out else torch.cat(results[0])

def one_hot(codes, k):
    """return a one-hot matrix where each code is represented as a 1"""
    nt, M = codes.shape
    tab = np.zeros((nt * M, k), dtype="float32")
    tab[np.arange(nt * M), codes.ravel()] = 1
    return tab.reshape(nt, M, k)

def compute_fixed_codebooks(xt, train_codes, k=256):
    """estimate fixed codebooks that minimize the reconstruction loss
    w.r.t. xt given the train_codes"""
    nt, M = train_codes.shape
    nt2, d = xt.shape
    assert nt2 == nt
    onehot_codes = one_hot(train_codes, k).reshape((nt, -1))
    codebooks, _, _, _ = np.linalg.lstsq(onehot_codes, xt, rcond=None)
    codebooks = codebooks.reshape((M, k, d))
    return codebooks

def reconstruct_from_fixed_codebooks(codes, codebooks):
    """reconstruct vectors from thier codes and the fixed codebooks"""
    M = codes.shape[1]
    assert codebooks.shape[0] == M
    for m in range(M):
        xi = codebooks[m, codes[:, m]]
        if m == 0:
            recons = xi
        else:
            recons += xi
    return recons