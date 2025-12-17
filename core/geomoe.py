import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils import weighted_8points, knn_point, get_knn_feats



class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None, use_bn=True, use_short_cut=True):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels

        self.use_short_cut=use_short_cut
        if use_short_cut:
            self.shot_cut = None
            if out_channels != channels:
                self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        if use_bn:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )
        else:
            self.conv = nn.Sequential(
                    nn.InstanceNorm2d(channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(channels, out_channels, kernel_size=1),
                    nn.InstanceNorm2d(out_channels, eps=1e-3),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    )

    def forward(self, x):
        out = self.conv(x)
        if self.use_short_cut:
            if self.shot_cut:
                out = out + self.shot_cut(x)
            else:
                out = out + x
        return out


class ConsenKnn(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.mlp_feats=nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1),
        )
        self.mlp_knn=nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1),
        )
        self.mlp_loc=nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, feats, knn_feats):
        nn_feats_diff = (feats - knn_feats).contiguous()      
        nn_feats_out  = self.mlp_feats(nn_feats_diff) + nn_feats_diff
        nn_feats_out  = nn_feats_out.transpose(1, 3)          # BCNK → BKNC

        feats_out     = self.mlp_knn(nn_feats_out) + nn_feats_out
        loc_feats     = self.mlp_loc(feats_out).transpose(1, 3)  # BKNC → BCN1
        return loc_feats

class LacalLayer(nn.Module):
    def __init__(self,channels,knn_dim):
        super().__init__()
        self.conv_down=nn.Conv2d(channels,knn_dim,1)
        self.knn_feats=ConsenKnn(knn_dim)
        self.conv_up=nn.Conv2d(knn_dim,channels,1)

    def forward(self, feats, idxs):
        feats_old = feats.clone()
        feats=self.conv_down(feats)
        nn_feats=get_knn_feats(feats,idxs)
        feats_knn=self.knn_feats(feats, nn_feats)
        feats = self.conv_up(feats_knn) + feats_old
        return feats 


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head, mode='full'):
        nn.Module.__init__(self)
        self.head = head
        self.mode = mode
        self.head_dim = channels // head
        if mode=='linear':
            self.feature_map = elu_feature_map
            self.eps = 1e-6

        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(inplace=True),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, x1, x2, kv_mask=None):
        # x1(q) attend to x2(k,v)
        batch_size = x1.shape[0]
        query, key, value = self.query_filter(x1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(x2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(x2).view(batch_size, self.head, self.head_dim, -1)

        if self.mode == 'full':
            QK = torch.einsum('bhdn,bhdm->bhnm', query, key)
            # set masked position to -1e6
            if kv_mask is not None:
                QK.masked_fill_(~(kv_mask[:, None, None, :]), float(-1e6))
            score = torch.softmax(QK / self.head_dim ** 0.5, dim = -1) # BHNM
            add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
            # assign_mat = torch.mean(torch.softmax(QK/self.head_dim**0.5,dim=-2),dim=1,keepdim=False).permute(0,2,1) # BMN
        elif self.mode == 'linear':
            # set masked position to zero
            if kv_mask is not None:
                key = key * kv_mask[:, None, None, :]
                value = value * kv_mask[:, None, None, :]
            Q = self.feature_map(query) # BHDN
            K = self.feature_map(key) # BHDM
            v_length = value.shape[-1] # BHVM
            value = value / v_length  # prevent fp16 overflow
            KV = torch.einsum("bhdm,bhvm->bhdv", K, value)
            Z = 1 / (torch.einsum("bhdn,bhd->bhn", Q, K.sum(dim=-1)) + self.eps)
            add_value = torch.einsum("bhdn,bhdv,bhn->bhvn", Q, KV, Z).reshape(batch_size, self.head_dim * self.head, -1) * v_length # B(HD)N
        else:
            raise KeyError

        add_value = self.mh_filter(add_value)
        x1_new = x1 + self.cat_filter(torch.cat([x1, add_value], dim=1))
        return x1_new


class diff_Pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))
    def forward(self, x):
        # x: b,f,n--->bfn1
        x = x.unsqueeze(3)
        embed = self.conv(x)  # b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)  # b,k,n *1
        # b,f,n @ b,n,k
        out = torch.bmm(x.squeeze(3), S.transpose(1, 2))
        return out # b,f,k
  
  
class MoeLayer(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        hidden_dim: int = None
    ):
        super().__init__()
        self.dim = dim
        self.E = num_experts
        self.k = num_experts_per_tok

        self.hidden_dim = 16
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, dim, bias=True),
            )
            for _ in range(self.E)
        ])

        self.gate =nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim,num_experts,1),
        )

    def forward(self, inputs: torch.Tensor):
        B, N, C = inputs.shape
        assert C == self.dim

        flat = inputs.reshape(B * N, C)

        logits_gate = self.gate(inputs.transpose(1, 2)).transpose(1, 2)
        logits = logits_gate.reshape(B * N, -1)            
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)  
        topk_w = F.softmax(topk_vals, dim=-1)                   

        weight = torch.zeros_like(logits)
        weight.scatter_(1, topk_idx, topk_w)

        expert_outs = [expert(flat) for expert in self.experts]   
        all_out = torch.stack(expert_outs, dim=1)                

        w = weight.unsqueeze(-1)                                
        out_flat = (all_out * w).sum(dim=1)                      
        out = out_flat.view(B, N, C)
        usage = weight.sum(dim=0) / (B * N)  # (E,)
        lb_loss = (usage ** 2).sum() * self.E

        return out.transpose(1, 2), lb_loss


class Recification(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        
        self.context = AttentionPropagation(channels, head)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.mlp = nn.Sequential(
            nn.Conv1d(channels,channels,1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels,1,1)
        )
        
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(inplace=True),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )


    def forward(self, x):
        B, C, N = x.shape  
        y = self.avg_pool(x.unsqueeze(-1)).view(B, C, 1)  
        weights = self.mlp(y)  # [B, C, 1]
        out_channel = x * weights  # [B, C, N]
        out_context = self.context(x, x)
        out = self.cat_filter(torch.cat([out_channel, out_context], dim=1))
        return out


class MoEGAT(nn.Module):
    def __init__(self, channels, head, pattern_num, num_experts, num_experts_per_tok):
        nn.Module.__init__(self)
        self.head = head
        self.channels = channels
        self.pattern_num = pattern_num
        self.subpattern = diff_Pool(channels, self.pattern_num)
        
        self.feed_forward1 = MoeLayer(
                channels,
                num_experts,
                num_experts_per_tok,
            )
        
        self.cluster = AttentionPropagation(channels, self.head)
        self.sub_rectify = Recification(channels, self.head)
        
        self.feed_forward2 = MoeLayer(
                channels,
                num_experts,
                num_experts_per_tok,
            )
        self.decluster = AttentionPropagation(channels, self.head)


    def forward(self, x, mask):
        # BCN1->BCN
        x = x.squeeze(3)
        load_balance_loss = 0
        
        sub_pattern = self.subpattern(x)
        
        # F-MoE
        out_moe, load_balance_loss1 = self.feed_forward1.forward(sub_pattern.transpose(1, 2)) 
        sub_pattern = out_moe + sub_pattern
        
        # Probabilistic Prior-Guided Decomposition
        sub_pattern = self.cluster(sub_pattern, x, mask)  
        
        # MoE-Enhanced Bi-Path Rectifier
        sub_pattern = self.sub_rectify(sub_pattern)
        out_moe, load_balance_loss2 = self.feed_forward2.forward(sub_pattern.transpose(1, 2)) #BNC
        sub_pattern = out_moe + sub_pattern
        x = self.decluster(x, sub_pattern) # BCN

        return x.unsqueeze(3), (load_balance_loss1+load_balance_loss2)/2 # BCN1


class GeoMoEblock(nn.Module):
    def __init__(self, channels, knn_dim, pattern_num, head, num_experts, num_experts_per_tok):
        nn.Module.__init__(self)
        self.local = LacalLayer(channels, knn_dim)
        self.moegat = MoEGAT(channels, head, pattern_num, num_experts, num_experts_per_tok) 
        
        self.prob_predictor=nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels,1,1),
        )

    def forward(self, d, xs, idxs, logits=None):
        if logits is not None:
            weights = torch.relu(torch.tanh(logits)) # BN
            mask = weights>0 # BN
        else:
            mask = None
        d = self.local(d, idxs)
        d, load_balance_loss = self.moegat(d, mask)
        
        # BCN1 -> BN1 -> BN
        logits = torch.squeeze(torch.squeeze(self.prob_predictor(d), 1), 2)
        e_hat = weighted_8points(xs, logits)
        return d, logits, e_hat, load_balance_loss


class GeoMoE(nn.Module):
    def __init__(self, config, use_gpu = True): # 
        super().__init__()
        self.knn_num = config.knn_num
        self.num_block = config.num_block

        self.geom_embed = nn.Sequential(nn.Conv2d(4, config.channels,1),\
                                        PointCN(config.channels))
        self.geomoeblock_list = nn.ModuleList()
        for k in range(config.num_block):
            self.geomoeblock_list.append(GeoMoEblock(config.channels, config.knn_num, config.pattern_num, config.head, config.num_experts, config.num_experts_per_tok))

    def forward(self, data):
        # B1NC
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        # B1NC->B2N1->BCN1
        input = data['xs'].transpose(1,3) # BCN1
        _, idxs = knn_point(self.knn_num, input[...,0].transpose(1, 2), input[...,0].transpose(1, 2))
        idxs = idxs.permute(0, 2, 1)
        
        x1, x2= input[:,:2,:,:], input[:,2:,:,:]
        motion = torch.cat([x1, x2-x1], dim = 1)
        feats = self.geom_embed(motion) # BCN1
        
        res_logits, res_e_hat = [], []
        logits = None
        load_balance_loss_all = 0
        for net in self.geomoeblock_list:
            feats, logits, e_hat, load_balance_loss = net(feats, data['xs'], idxs, logits) # BCN BN
            res_logits.append(logits), res_e_hat.append(e_hat)
            load_balance_loss_all += load_balance_loss
        
        return res_logits, res_e_hat, load_balance_loss_all/self.num_block



