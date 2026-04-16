import faiss
import torch

def get_nearestneighbors_faiss(xb, xq, k, device='cpu', exact=True):
    if exact:
        index = faiss.IndexFlatL2(xb.shape[-1])
    else:
        index = faiss.index_factory(xb.shape[-1], 'HNSW32')
        index.hnsw.efSearch = 64
    index.add(xb)
    D, I = index.search(xq, k=k)
    return I

def cdist2(A, B):
    return  (A.pow(2).sum(1, keepdim = True)
             - 2 * torch.mm(A, B.t())
             + B.pow(2).sum(1, keepdim = True).t())

def top_dist(A, B, k):
    return cdist2(A, B).topk(k, dim=1, largest=False, sorted=True)[1]

def get_nearestneighbors_torch(xq, xb, k, device='cpu'):

    assert device in ["cpu", "cuda"]
    xb, xq = torch.from_numpy(xb), torch.from_numpy(xq)
    xb, xq = xb.to(device), xq.to(device)
    bs = 500
    I = torch.cat([top_dist(xq[i*bs:(i+1)*bs], xb, k)
                   for i in range(xq.size(0) // bs)], dim=0)
    I = I.cpu()
    return I.numpy()

get_nearestneighbors = get_nearestneighbors_faiss