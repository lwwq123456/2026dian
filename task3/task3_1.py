import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        
        variance = x_centered.pow(2).mean(dim=-1, keepdim=True) 
        return x_centered * torch.rsqrt(variance + self.eps) * self.weight


class GatedDeltaRuleRecurrent(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.alpha_proj = nn.Linear(d_model, 1)
        self.beta_proj = nn.Linear(d_model, 1)

        self.q_conv = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)
        self.k_conv = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)
        self.v_conv = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)

        self.out_norm = ZeroCenteredRMSNorm(d_model)
        self.output_gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d = x.shape
        
        q_l, k_l, v_l = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q_c = self.q_conv(q_l.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        k_c = self.k_conv(k_l.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        v_c = self.v_conv(v_l.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        
        q = F.normalize(F.silu(q_c), p=2, dim=-1)
        k = F.normalize(F.silu(k_c), p=2, dim=-1)
        v = F.silu(v_c)
          
        alpha = torch.sigmoid(self.alpha_proj(x))  
        beta = torch.sigmoid(self.beta_proj(x))

        output = []
        S = torch.zeros(batch, d, d, device=x.device)
        I = torch.eye(d, device=x.device).unsqueeze(0)

        for t in range(seq_len):  
            qt, kt, vt = q[:, t, :], k[:, t, :], v[:, t, :]
            at, bt = alpha[:, t, :], beta[:, t, :]
            
            kt_col, kt_row = kt.unsqueeze(2), kt.unsqueeze(1)
            vt_col = vt.unsqueeze(2)
              
            # S_t = S_{t-1} * (a_t * (I - b_t * k_t * k_t^T)) + b_t * v_t * k_t^T
            forget_term = at.unsqueeze(-1) * (I - bt.unsqueeze(-1) * torch.bmm(kt_col, kt_row))
            new_knowledge = bt.unsqueeze(-1) * torch.bmm(vt_col, kt_row)
            
            S = torch.bmm(S, forget_term) + new_knowledge
            ot = torch.bmm(S, qt.unsqueeze(2)).squeeze(2) 
            output.append(ot) 

        gdn_out = torch.stack(output, dim=1)
         
        gdn_out = self.out_norm(gdn_out)
        out_gate = F.silu(self.output_gate_proj(x))
        return self.out_proj(gdn_out * out_gate)
