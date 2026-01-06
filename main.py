import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Define RMSNorm as per standard implementation
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

# mHCLayer implementation based on the paper details
class mHCLayer(nn.Module):
    """
    mHC Layer: Manifold-Constrained Hyper-Connections, replacing standard residual connections.
    Based on the exact formulations from the DeepSeek mHC paper.
    """
    def __init__(self, dim, expansion_rate=4, sinkhorn_iter=20, gate_init=0.01, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.n = expansion_rate
        self.sinkhorn_iter = sinkhorn_iter
        self.eps = eps
        
        # RMSNorm for the flattened hidden state
        self.rms = RMSNorm(self.n * self.dim)
        
        # Gating factors
        self.alpha_pre = nn.Parameter(torch.tensor(gate_init))
        self.alpha_post = nn.Parameter(torch.tensor(gate_init))
        self.alpha_res = nn.Parameter(torch.tensor(gate_init))
        
        # Projection weights (using Linear for standard init)
        self.phi_pre = nn.Linear(self.n * self.dim, self.n, bias=False)
        self.phi_post = nn.Linear(self.n * self.dim, self.n, bias=False)
        self.phi_res = nn.Linear(self.n * self.dim, self.n ** 2, bias=False)
        
        # Static biases
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))
        
        # Initialize biases to approximate identity mapping
        with torch.no_grad():
            # For H_pre ~ (1, 0, ..., 0)
            self.b_pre.fill_(-10.0)
            self.b_pre[0] = 10.0
            # For H_post ~ (1, 0, ..., 0) since 2 * sigmoid
            self.b_post.fill_(-10.0)
            self.b_post[0] = 0.0  # sigmoid(0) = 0.5, 2*0.5=1
            # For H_res ~ eye
            self.b_res.fill_(-10.0)
            self.b_res.diagonal().fill_(10.0)

    def sinkhorn_knopp(self, mat):
        """Sinkhorn-Knopp iteration to project to doubly stochastic matrix"""
        # mat: (BT, n, n)
        mat = torch.exp(mat)
        for _ in range(self.sinkhorn_iter):
            mat = mat / mat.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            mat = mat / mat.sum(dim=-2, keepdim=True).clamp(min=self.eps)
        return mat

    def forward(self, x, func):
        # x: (B, T, C)
        B, T, C = x.shape
        BT = B * T
        
        # Expand to multi-stream hidden: (B, T, n, C), first stream is x, others 0
        hidden = torch.zeros(B, T, self.n, C, dtype=x.dtype, device=x.device)
        hidden[:, :, 0, :] = x
        
        # Flatten: (BT, n*C)
        flattened = hidden.view(BT, self.n * C)
        
        # RMSNorm
        x_prime = self.rms(flattened)
        
        # Compute tilde mappings
        tilde_pre = self.alpha_pre * self.phi_pre(x_prime) + self.b_pre
        tilde_post = self.alpha_post * self.phi_post(x_prime) + self.b_post
        tilde_res = self.alpha_res * self.phi_res(x_prime).view(BT, self.n, self.n) + self.b_res
        
        # Final constrained mappings
        H_pre = torch.sigmoid(tilde_pre)  # (BT, n)
        H_post = 2.0 * torch.sigmoid(tilde_post)  # (BT, n)
        H_res = self.sinkhorn_knopp(tilde_res)  # (BT, n, n)
        
        # Project input to func: H_pre @ hidden (as row vector mult)
        # H_pre.unsqueeze(1): (BT, 1, n), hidden.view(BT, n, C): (BT, n, C)
        pre_x = torch.bmm(H_pre.unsqueeze(1), hidden.view(BT, self.n, C)).squeeze(1)  # (BT, C)
        
        # Apply func (e.g., attention or FFN)
        func_out = func(pre_x.view(B, T, C))  # (B, T, C)
        func_out_bt = func_out.view(BT, C)  # (BT, C)
        
        # Compute add term: H_post^T @ func_out (column @ row)
        # H_post.unsqueeze(2): (BT, n, 1), func_out_bt.unsqueeze(1): (BT, 1, C)
        add_term = torch.bmm(H_post.unsqueeze(2), func_out_bt.unsqueeze(1))  # (BT, n, C)
        
        # Compute res term: H_res @ hidden
        res_term = torch.bmm(H_res, hidden.view(BT, self.n, C))  # (BT, n, C)
        
        # Combine
        out_expanded = res_term + add_term  # (BT, n, C)
        
        # Output the main stream (first stream)
        out = out_expanded[:, 0, :].view(B, T, C)
        
        return out

# Modified to use mHCLayer for both attention and FFN residuals.

class GPTConfig:
    """ Config class from nanoGPT """
    def __init__(self, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=True, block_size=1024):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.block_size = block_size

class LayerNorm(nn.Module):
    """ LayerNorm from nanoGPT """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ CausalSelfAttention from nanoGPT (simplified, no flash) """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Modified Block using mHCLayer for both residuals
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        # mHC layers
        self.mhc_att = mHCLayer(config.n_embd)
        self.mhc_ffn = mHCLayer(config.n_embd)

    def forward(self, x):
        # mHC for attention
        x = self.mhc_att(self.ln_1(x), self.attn)
        # mHC for FFN
        x = self.mhc_ffn(self.ln_2(x), self.mlp)
        return x

# Simple GPT model using the modified Block
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Share weights
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

# Demo usage: Create a small model
if __name__ == "__main__":
    config = GPTConfig(vocab_size=50304, n_layer=2, n_head=2, n_embd=64, dropout=0.1, block_size=128)
    model = GPT(config)
    print(model)

    # Dummy input
    idx = torch.randint(0, config.vocab_size, (4, 32))  # batch_size=4, seq_len=32
    logits, loss = model(idx)
    print(f"Logits shape: {logits.shape}")
```
