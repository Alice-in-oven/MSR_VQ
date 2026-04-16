import os
import faiss
import torch
import struct
import numpy as np
import pandas as pd
def geolife_embedding(data_root, train_size, test_size):
    '''
    xt = np.load(os.path.join(data_root, 'train_dtw_feature_test'),allow_pickle=True)
    xb = np.load(os.path.join(data_root, 'base_dtw_feature_test'),allow_pickle=True)
    xq = np.load(os.path.join(data_root, 'query_dtw_feature_test'),allow_pickle=True)
    #xt = np.concatenate((xt, xb,xq),axis=0)
    #xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    '''
    embs = np.load(os.path.join(data_root, 'dtw_feature_test'),allow_pickle=True)
    xt = embs[:9000]
    xq = embs[9000:10000]
    xb = embs[10000:]
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt
def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def read_fbin(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    dim = x[1]
    return x[2:].view('float32').reshape(-1, dim)

def parquet_read(path):
    df = pd.read_parquet(path)
    x = np.stack(df['emb'].infer_objects(), axis=0)
    return x

def read_bin(path):
    fq = open(path, 'rb')
    nvecs = struct.unpack('i', fq.read(4))[0]
    dim = struct.unpack('i', fq.read(4))[0]
    vectors = np.frombuffer(fq.read(nvecs * dim * 4), dtype=np.float32).reshape((nvecs, dim))
    return vectors

def parquet_gt(path):
    df = pd.read_parquet(path)
    gt = np.stack(df['neighbors_id'].infer_objects(), axis=0)
    return gt

def generate_gt_nn(xb, xq, k=100):
    index = faiss.IndexFlatL2(xb.shape[-1])
    index.add(xb)
    _, gt = index.search(xq, k=k)
    return gt

def load_sift1m(data_root, train_size=5*10**5, test_size=10**6, mode='train'):
    xt = mmap_fvecs(data_root + '/sift_learn.fvecs')[:train_size]
    xb = mmap_fvecs(data_root + '/sift_base.fvecs')[:test_size]
    xq = mmap_fvecs(data_root + '/sift_query.fvecs')
    gt = generate_gt_nn(xb, xq)
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    return xt, xb, xq, gt

def load_gist1m(data_root, train_size=5*10**5, test_size=10**6):
    xt = mmap_fvecs(data_root + '/gist_learn.fvecs')[:train_size]
    xb = mmap_fvecs(data_root + '/gist_base.fvecs')[:test_size]
    xq = mmap_fvecs(data_root + '/gist_query.fvecs')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_bigann1m(data_root, train_size=5*10**5, test_size=10**6):
    xt = mmap_bvecs(data_root + '/bigann_learn.bvecs')[:train_size]
    xb = mmap_bvecs(data_root + '/bigann_base.bvecs')[:test_size]
    xq = mmap_bvecs(data_root + '/bigann_query.bvecs')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

def load_deep1m(data_root, train_size=5*10**5, test_size=10**6):
    xt = read_fbin(data_root + '/learn.350M.fbin')[:train_size]
    xb = read_fbin(data_root + '/base.1B.fbin')[:test_size]
    xq = read_fbin(data_root + '/query.public.10K.fbin')
    xt, xb, xq = sanitize(xt), sanitize(xb), sanitize(xq)
    gt = generate_gt_nn(xb, xq)
    return xt, xb, xq, gt

class VectorDataset:
    def __init__(self, dataset='sift1m', data_path=None, normalize=True, train_size=10**5, test_size=10**6, device='cpu'):
        assert data_path is not None, "data_path must be specified"
        self.dataset = dataset
        self.data_path = data_path
        self.normalize = normalize

        xt, xb, xq, gt = READER[dataset](data_path, train_size, test_size)
        
        self.xt = torch.from_numpy(xt).to(device)
        print("Training set size:", self.xt.shape)
        self.xb = torch.from_numpy(xb).to(device)
        self.xq = torch.from_numpy(xq).to(device)
        self.gt = gt

        if normalize:
            mean_norm = self.xt.norm(p=2, dim=-1).mean().item()
            self.xt = self.xt / mean_norm
            self.xb = self.xb / mean_norm
            self.xq = self.xq / mean_norm


READER = {
    'sift1m': load_sift1m,
    'gist1m': load_gist1m,
    'bigann1m': load_bigann1m,
    'deep1m': load_deep1m,
    'geolife_embedding': geolife_embedding
}
    
