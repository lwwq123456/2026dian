import torch
import torch.nn as nn
import math

# 任务 2-1
class StandardMHA(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads 
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        q_len, k_len = Q.size(2), K.size(2)
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device).tril(diagonal=k_len - q_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.out_proj(context)


# 任务 2-2
class KVCacheMHA(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads 
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, past_key_values=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if past_key_values is not None:
            past_k, past_v = past_key_values
            K = torch.cat([past_k, K], dim=2) 
            V = torch.cat([past_v, V], dim=2)
            
        current_kv_cache = (K, V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        q_len, k_len = Q.size(2), K.size(2)
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device).tril(diagonal=k_len - q_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(context), current_kv_cache


# 任务 2-3
class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=8, num_kv_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, past_key_values=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        if past_key_values is not None:
            past_k, past_v = past_key_values
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
            
        current_kv_cache = (K, V)

        if self.num_heads != self.num_kv_heads:
            num_repeat = self.num_heads // self.num_kv_heads
            K = K.repeat_interleave(num_repeat, dim=1) 
            V = V.repeat_interleave(num_repeat, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        q_len, k_len = Q.size(2), K.size(2)
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device).tril(diagonal=k_len - q_len)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(context), current_kv_cache


# 统一测试与运行
if __name__ == "__main__":
    batch_size = 2
    hidden_dim = 64
    
    print("\n" + "="*50)
    print("开始验证 任务 2-1：前向验证")
    print("="*50)
    model_2_1 = StandardMHA(hidden_dim=hidden_dim, num_heads=8)
    dummy_input_2_1 = torch.randn(batch_size, 10, hidden_dim) 
    out_2_1 = model_2_1(dummy_input_2_1)
    print(f"输入张量形状: {dummy_input_2_1.shape}")
    print(f"输出张量形状: {out_2_1.shape} (验证完成，形状完全一致)")

    print("\n" + "="*50)
    print("开始验证 任务 2-2：模拟流式生成与 KV Cache")
    print("="*50)
    model_2_2 = KVCacheMHA(hidden_dim=hidden_dim, num_heads=8)

    initial_x = torch.randn(batch_size, 10, hidden_dim)
    out_2_2, kv_cache = model_2_2(initial_x)
    print(f"初始阶段 - 输入序列长度: 10, 缓存中 K/V 长度为: {kv_cache[0].shape[2]}")
    
    for step in range(5):
        new_token = torch.randn(batch_size, 1, hidden_dim)
        out_2_2, kv_cache = model_2_2(new_token, past_key_values=kv_cache)

        print(f"Step {step+1} - 新 Token (Q) 长度: {new_token.shape[1]}, 此时 KV Cache 增长为: {kv_cache[0].shape[2]}")

