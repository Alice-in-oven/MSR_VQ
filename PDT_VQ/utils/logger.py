import os
import time
import torch
import json
from collections import OrderedDict


class Logger:
    def __init__(self, args):
        self.args = args
        supervision = "ms_sup" if args.ms_sup else "ss_sup"
        exp_sign = '{}-{}-{}_transform-M{}-K{}-D{}-S{}-H{}-L{}-{}_init-rerank_{}-{}'.format(args.dataset, args.vq_type, args.trans_type, args.M, args.K, args.d_hidden, args.steps, args.heads, args.L, args.codebook_init, args.re_rank, supervision)
        
        if self.args.log:
            self.log_path = generate_path(os.path.join(args.log_path, exp_sign))
            self.checkpoint_path = generate_path(os.path.join(args.log_path, exp_sign, 'checkpoint'))
            arg_dict = vars(args)
            with open(os.path.join(self.log_path, 'config.json'), 'w') as f:
                json.dump(arg_dict, f)

        self.epoch = 0
        self.sep = '-'
    
    def log(self, log_str):
        print(log_str, sep='')
        if self.args.log:
            with open(os.path.join(self.log_path, 'log.txt'), 'a+') as f:
                f.write(log_str+'\n')
    
    def log_model(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        log_str = 'model info'.center(100, self.sep) + '\n'
        log_str += f'structure: transform: {self.args.trans_type} | vq: {self.args.vq_type} | parameters: {total_num}\n'
        log_str += f'd_in: {model.d} | d_hidden: {model.d_hidden} | M: {model.M} | K: {model.K} | S: {self.args.steps} | H: {self.args.heads} | L: {self.args.L} | step_norm: {not self.args.no_step_norm} | head_norm: {not self.args.no_head_norm} |'
        self.log(log_str)

    def log_dataset(self, dataset):
        log_str = 'dataset info'.center(100, self.sep) + '\n'
        log_str += f'dataset: {dataset.dataset} | data_path: {os.path.join(dataset.data_path)} \n'
        log_str += f'train size: {len(dataset.xt)} | test size: {len(dataset.xb)} | query size: {len(dataset.xq)} | dimension: {dataset.xt.shape[-1]} | normalize: {self.args.normalize}'
        self.log(log_str)
        self.iterations = len(list(range(0, len(dataset.xt), self.args.batch_size)))

    def log_train(self, epoch=None, lr=None, loss=None, val_mse=None, is_head=False):
        if is_head:
            log_str = f'start training epochs={self.args.epochs:03d}'.center(100, self.sep)
        else:
            log_str = f'[Epoch: {epoch:03d}/{self.args.epochs} Lr: {lr:.6f}] | Loss: {loss:.6f} | Val_MSE: {val_mse:.6f}'
        self.log(log_str)

    def log_eval(self, eval_results, epoch=None):
        if epoch is None:
            epoch = self.epoch
        log_str = f'eval results epoch {epoch}'.center(100, self.sep) + '\n'
        for L, recalls in eval_results:
            log_str += f'| L={L}\t | '
            for k, v in recalls.items():
                log_str += f'{k}: {v:.4f} | '
            log_str += '\n'
        self.log(log_str)

    def save_checkpoint(self, model, is_best=False):
        if self.args.log:
            if is_best:
                torch.save(model.state_dict(), os.path.join(self.checkpoint_path, 'model-best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(self.checkpoint_path, 'model-last.pth'))
    

def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path