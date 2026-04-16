import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .dpq import DPQ
from .drq import DRQ
from .qinco import QINCo
from .trans_layers import DefaultTransLayer, OrthogonalTrans, MLPTrans, MultiStepDistributionTrans
from utils.loss import distance_loss, uniform_loss


class PDT(nn.Module):
    def __init__(self, d, args) -> None:
        super().__init__()
        """
        d: original vector dimension
        d_hidden: transformed vector dimension
        M: number of codes
        K: bits of each code
        """
        self.args = args
        self.M = args.M
        self.K = args.K
        self.d, self.d_hidden = d, args.d_hidden
        self.step_norm = not args.no_step_norm
        self.head_norm = not args.no_head_norm
        self.ms_sup = args.ms_sup
        
        if self.args.trans_type == 'no':
            self.trans = DefaultTransLayer(self.d)
        elif self.args.trans_type == 'orth':
            self.trans = OrthogonalTrans(self.d)
        elif self.args.trans_type == 'mlp':
            self.trans = MLPTrans(self.d, self.d_hidden, args.steps)
        elif self.args.trans_type == 'msd':
            self.trans = MultiStepDistributionTrans(self.d, self.d_hidden, self.M, args.steps, args.heads, self.step_norm, self.head_norm)
        else:
            raise NotImplementedError

        if self.args.vq_type == 'dpq':
            print('using deep product quantizer')
            self.vq = DPQ(self.d, self.M, self.K, args.codebook_init)
        elif self.args.vq_type == 'drq':
            print('using deep residual quantizer')
            self.vq = DRQ(self.d, self.M, self.K, args.codebook_init)
        elif self.args.vq_type == 'qinco':
            print('using qinco quantizer')
            self.vq = QINCo(self.d, self.M, self.K, args.qinco_h, args.qinco_L, args.codebook_init, identity_init=getattr(args, "qinco_identity_init", False))
        else:
            raise NotImplementedError
        
        self.use_normalization = True # 强制开启归一化
    
        # [新增] 定义结构损失的动态权重调度 (Rank权重, MSE权重)
        # 逻辑复制自 ConvTraj，根据 Epoch 调整侧重点
        total_epochs = args.epoch_num
        step = total_epochs // 5
        self.sem_schedule = {
            step * 0: (0, 10),    # 初期重 MSE
            step * 1: (10, 10),
            step * 2: (10, 1),
            step * 3: (5, 0.1),
            step * 4: (1, 0.01),  # 后期重 Ranking
        }
        self.sem_mlp_r = 1.0 # 当前 Rank 权重
        self.sem_m = 1.0     # 当前 MSE 权重
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if epoch in self.sem_schedule:
            self.sem_mlp_r, self.sem_m = self.sem_schedule[epoch]
    
    def init_codebook(self, x, resume=None):
        x = self.encode(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        self.vq.init_codebook(x, resume)

    def init_transform(self, resume):
        if resume is not None:
            state_dict = torch.load(resume)
            new_state_dict = {}
            for k, v in self.state_dict().items():
                if 'trans' in k:
                    new_state_dict[k] = state_dict[k]
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict)

    def _encode_last(self, x, out_step=None):
        x_enc = self.encode(x, out_step=out_step)
        if isinstance(x_enc, (list, tuple)):
            x_enc = x_enc[-1]
        return x_enc

    def _codebook_uniform_loss(self, reference):
        quantizer = getattr(self.vq, "quantizer", None)
        if isinstance(quantizer, nn.ModuleList):
            losses = [uniform_loss(q.codebook) for q in quantizer if hasattr(q, "codebook")]
            if losses:
                return torch.stack(losses).mean()
        elif quantizer is not None and hasattr(quantizer, "codebook"):
            return uniform_loss(quantizer.codebook)
        elif hasattr(self.vq, "codebook0") and hasattr(self.vq, "steps"):
            codebooks = [self.vq.codebook0.weight]
            for step in self.vq.steps:
                if hasattr(step, "codebook"):
                    codebooks.append(step.codebook.weight)
            if codebooks:
                stacked = torch.stack(codebooks, dim=0)
                return uniform_loss(stacked)
        return reference.new_tensor(0.0)
    
    def encode(self, x, out_step=None):
        return self.trans.encode(x, out_step)
    
    def get_codes(self, x):
        return self.vq.get_codes(x)
    
    def reconstruction(self, codes):
        return self.vq.reconstruction(codes)
    
    def decode(self, x, out_step=None):
        return self.trans.decode(x, out_step)

    def soft_quantize(self, x):
        x_enc = self._encode_last(x, out_step=self.args.steps)
        x_recon_vq, soft_codes, _ = self.vq(x_enc)
        x_recon = self.decode(x_recon_vq, out_step=self.args.steps)[-1]

        codebook_loss = F.mse_loss(x_recon_vq, x_enc.detach())
        commitment_loss = F.mse_loss(x_recon_vq.detach(), x_enc)
        vq_commitment = codebook_loss + 0.25 * commitment_loss

        entropy_reg = x_enc.new_tensor(0.0)
        if torch.is_tensor(soft_codes) and soft_codes.dim() == 3 and soft_codes.dtype.is_floating_point:
            avg_probs = soft_codes.mean(dim=0).clamp_min(1e-8)
            entropy = -(avg_probs * avg_probs.log()).sum(dim=-1).mean()
            entropy_reg = x_enc.new_tensor(math.log(float(soft_codes.shape[-1]))) - entropy
        elif torch.is_tensor(soft_codes) and soft_codes.dim() == 2 and not soft_codes.dtype.is_floating_point:
            one_hot_codes = F.one_hot(soft_codes.long(), num_classes=self.K).float()
            avg_probs = one_hot_codes.mean(dim=0).clamp_min(1e-8)
            entropy = -(avg_probs * avg_probs.log()).sum(dim=-1).mean()
            entropy_reg = x_enc.new_tensor(math.log(float(self.K))) - entropy

        return x_recon, {
            "commitment_loss": vq_commitment,
            "entropy_loss": entropy_reg,
            "uniform_loss": self._codebook_uniform_loss(x_enc),
        }

    def _normalize_dist(self, tensor):
        """[新增] 距离归一化：解决 DTW 和 向量距离 量纲不一致的问题"""
        return tensor / (tensor.mean() + 1e-8)
    def forward_test(self, x):
        x_enc = self._encode_last(x, out_step=self.args.steps)
        x_recon_vq, _, side_output = self.vq(x_enc)
        x_recon = self.decode(x_recon_vq, out_step=self.args.steps)[-1]
        loss = torch.norm(x_recon - x, 2, dim=-1) + distance_loss(x_recon, x)
        return loss
    def inference(self, x):
        x_enc = self._encode_last(x)
        x_recon_vq, _, side_output = self.vq(x_enc)
        x_recon = self.decode(x_recon_vq)[-1]
        loss = torch.norm(x_recon - x, 2, dim=-1) + distance_loss(x_recon, x)
        return x_recon,loss
    def forward_train(self, anchor,pos,neg,pn_dist,ap_dist,an_dist):
        self.set_epoch(self.current_epoch)
        if not self.ms_sup:
            anc_emb, anc_loss = self.inference(anchor)
            pos_emb, pos_loss = self.inference(pos)
            neg_emb, neg_loss = self.inference(neg)
            d_ap = torch.norm(anc_emb - pos_emb, p=2, dim=1)
            d_an = torch.norm(anc_emb - neg_emb, p=2, dim=1)
            d_pn = torch.norm(pos_emb - neg_emb, p=2, dim=1)

            d_ap_norm = self._normalize_dist(d_ap)
            d_an_norm = self._normalize_dist(d_an)
            d_pn_norm = self._normalize_dist(d_pn)
            gt_ap_norm = self._normalize_dist(ap_dist)
            gt_an_norm = self._normalize_dist(an_dist)
            gt_pn_norm = self._normalize_dist(pn_dist)

            threshold = gt_an_norm - gt_ap_norm
            rank_loss = torch.nn.functional.relu(d_ap_norm - d_an_norm + threshold)
            mse_loss = (d_ap_norm - gt_ap_norm) ** 2 + \
                       (d_an_norm - gt_an_norm) ** 2 + \
                       (d_pn_norm - gt_pn_norm) ** 2
            recon_loss = anc_loss + pos_loss + neg_loss
            total_loss = recon_loss + self.sem_mlp_r * rank_loss + self.sem_m * torch.sqrt(mse_loss + 1e-12)

            return total_loss

        tau = 2
        norm = torch.exp(torch.as_tensor([i / tau for i in range(self.args.steps)])).sum().to(anchor.device)
        xs_enc = self.encode(anchor, out_step=self.args.steps)
        x_recons = []
        side_outputs = []
        for i in range(self.args.steps):
            x_recon_vq_i, _, side_output_i = self.vq(xs_enc[i])
            x_recon_i = self.decode(x_recon_vq_i, out_step=i+1)[-1]
            x_recons.append(x_recon_i)
            side_outputs.append(side_output_i)
        loss = torch.zeros(len(anchor)).to(anchor.device)
        for k, x_recon in enumerate(x_recons):
            weight = torch.exp(torch.tensor(k / tau).to(anchor.device))
            loss += weight / norm * (torch.norm(x_recon - anchor, 2, dim=-1) + distance_loss(x_recon, anchor))
        return loss

    def forward(self, anchor,pos,neg,pn_dist,ap_dist,an_dist):
        if not self.training:
            return self.forward_test(anchor)
        else:
            return self.forward_train(anchor,pos,neg,pn_dist,ap_dist,an_dist)
        
        
