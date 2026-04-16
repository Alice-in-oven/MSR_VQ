import argparse
import sys
import torch
import numpy as np
import os
from torch.optim import AdamW
from dataset import VectorDataset
from model import PDT
from utils import train_one_epoch, eval, Logger, Scheduler, compute_mse

def get_args(argv=None):
    parser = argparse.ArgumentParser()
    group_dataset = parser.add_argument_group('Dataset Options')
    group_dataset.add_argument("--dataset", type=str, default='sift1m', choices=['geolife_embedding','sift1m', 'gist1m', 'bigann1m', 'deep1m'],)
    group_dataset.add_argument("--data_root", type=str, default='/data3/menghaotian/Traj_sim/PDT-VQ/data/sift1m', help='The root directory of the dataset')
    group_dataset.add_argument("--normalize", type=bool, default=True, help='Whether to normalize the dataset vectors')
    group_dataset.add_argument("--train_size", type=int, default=5*10**5, help='The number of training vectors')
    group_dataset.add_argument("--test_size", type=int, default=10**6, help='The number of test vectors')
    group_dataset.add_argument("--val_size", type=int, default=10**5, help='The number of validation vectors')

    group_model = parser.add_argument_group('Model Options')
    group_model.add_argument("--d_hidden", type=int, default=256, help='The hidden dimension of the model')
    group_model.add_argument("--M", type=int, default=4, help='The number of codes each vector')
    group_model.add_argument("--K", type=int, default=2**6, help='The number of centriods in each code')
    
    group_model.add_argument("--L", type=int, default=1000, help='The search length in the evaluation process')
    group_model.add_argument("--steps", type=int, default=1)
    group_model.add_argument("--heads", type=int, default=1)
    group_model.add_argument("--trans_type", type=str, default='msd', choices=['no', 'orth', 'mlp', 'msd'])
    group_model.add_argument("--vq_type", type=str, default='dpq', choices=['dpq', 'drq', 'qinco'])
    group_model.add_argument("--no_step_norm", action='store_true')
    group_model.add_argument("--no_head_norm", action='store_true')
    group_model.add_argument("--codebook_init", type=str, default='uniform', choices=['uniform', 'faiss', 'resume'])
    group_model.add_argument("--re_rank", type=bool, default=True)
    # Parameters of QINCo
    group_model.add_argument("--qinco_h", type=int, default=256)
    group_model.add_argument("--qinco_L", type=int, default=1)
    group_model.add_argument("--ms_sup", type=bool, default=False)

    group_train = parser.add_argument_group('Training Options')
    group_train.add_argument("--batch_size", type=int, default=1024)
    group_train.add_argument("--epoch_num", type=int, default=100)
    group_train.add_argument("--lr", type=float, default=1e-3)
    group_train.add_argument("--resume", type=str, default=None)
    group_train.add_argument("--log", type=bool, default=True)
    group_train.add_argument("--log_path", type=str, default='./log')
    
    group_computation = parser.add_argument_group('Computation Options')
    group_computation.add_argument("--device", type=str, default='cuda:2')
    group_computation.add_argument("--seed", type=int, default=123456)

    args = parser.parse_args(argv)
    args.epochs = args.epoch_num
    return args

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not os.path.exists(args.log_path) and args.log:
        os.makedirs(args.log_path)
    log = Logger(args)
    
    # load dataset
    dataset = VectorDataset(args.dataset, args.data_root, args.normalize, args.train_size, args.test_size, args.device)
    log.log_dataset(dataset)

    # load model
    model = PDT(dataset.xt.shape[-1], args)
    model.to(args.device)
    log.log_model(model)

    # for initialize of pq parameters
    log.log('initialize the codebook centers...')
    with torch.no_grad():
        model.init_transform(args.resume)
        model.init_codebook(dataset.xt, args.resume)

    # init training process
    scheduler = Scheduler(args.lr)

    min_loss = compute_mse(model, dataset.xb[:args.val_size], args)
    best_epoch = 0
    log.log_train(is_head=True)
    for epoch in range(args.epochs):
        lr = scheduler.lr   # change the lr according to the scheduler
        optimizer = AdamW(model.parameters(), lr=lr)
        loss = train_one_epoch(model, dataset.xt, optimizer, args)
        loss_val = compute_mse(model, dataset.xb[:args.val_size], args)
        log.log_train(epoch, lr, loss, loss_val)
        if loss_val < min_loss:
            log.log('Updating the checkpoint according to the validation results...')
            min_loss = loss_val
            best_epoch = epoch
            log.save_checkpoint(model, True)

        scheduler.append_loss(loss_val)
        if scheduler.should_stop():
            break
    
    log.log('Evaluation...')
    model.load_state_dict(torch.load(os.path.join(log.checkpoint_path, 'model-best.pth')))
    model.eval()
    recalls = eval(model, dataset.xt, dataset.xb, dataset.xq, dataset.gt, args)
    log.log('best eval result:')
    log.log_eval(recalls, best_epoch)

if __name__ == '__main__':
    args = get_args()
    main(args)
