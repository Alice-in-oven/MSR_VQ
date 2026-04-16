import argparse
import torch
import json
import os

from dataset import VectorDataset
from model import PDT
from utils import eval
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()

    with open(os.path.join(args.exp_path, 'config.json'), 'r') as f:
        exp_args_dict = json.load(f)
    exp_args = argparse.Namespace(**exp_args_dict)

    dataset = VectorDataset(exp_args.dataset, exp_args.data_root, exp_args.normalize, exp_args.train_size, exp_args.test_size, args.device)
    model = PDT(dataset.xt.shape[-1], exp_args)
    model.load_state_dict(torch.load(os.path.join(args.exp_path, 'checkpoint', 'model-best.pth')))
    model.to(args.device)
    model.eval()

    recalls = eval(model, dataset.xt, dataset.xb, dataset.xq, dataset.gt, exp_args)
    for recall in recalls:
        print(recall)
    

    
