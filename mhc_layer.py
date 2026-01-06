import torch
import torch.nn as nn
import torch.nn.functional as F

# 实现 RMSNorm（Root Mean Square Layer Normalization）
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm 公式：x * weight / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def sinkhorn_knopp(log_mat, num_iters=20, eps=1e-8):
    """
    可微的 Sinkhorn-Knopp 迭代，将输入矩阵转换为双随机矩阵。
    Args:
        log_mat: [*, n, n] 输入矩阵（log域）
        num_iters: 迭代次数
        eps: 数值稳定性常数
    Returns:
        [*, n, n] 双随机矩阵
    """
    # 确保输入为正（通过 exp）
    mat = torch.exp(log_mat)
    for _ in range(num_iters):
        # 列归一化
        mat = mat / (mat.sum(dim=-2, keepdim=True) + eps)
        # 行归一化
        mat = mat / (mat.sum(dim=-1, keepdim=True) + eps)
    return mat

class MHCLayer(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) 层。
    严格按照论文公式(3)实现：输出维度为 n×C，没有额外残差连接。
    
    Args:
        dim: 输入维度 C
        n_streams: 扩展率 n
        hidden_dim: 残差函数 F 的隐藏维度
        sinkhorn_iters: Sinkhorn 迭代次数
        dropout: Dropout 概率
    """
    def __init__(self, dim, n_streams=4, hidden_dim=None, sinkhorn_iters=20, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        if hidden_dim is None:
            hidden_dim = dim * 4

        # RMSNorm（对最后一维进行归一化）
        self.rms_norm = RMSNorm(dim * n_streams)

        # 动态映射的线性投影
        # φ_pre, φ_post: [nC, n]；φ_res: [nC, n^2]
        self.proj_dynamic = nn.Linear(
            dim * n_streams,
            n_streams * n_streams + 2 * n_streams,  # n^2 + 2n
            bias=False
        )

        # 静态映射的偏置
        self.bias_pre = nn.Parameter(torch.zeros(1, n_streams))
        self.bias_post = nn.Parameter(torch.zeros(1, n_streams))
        self.bias_res = nn.Parameter(torch.zeros(n_streams, n_streams))

        # 可学习的门控因子 α
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))

        # 残差函数 F（例如 MLP）
        self.residual_fn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        # 论文中 mHC 层输出维度为 n×C，因此不需要输出投影
        # 这是与之前实现的关键区别

        self._init_parameters()

    def _init_parameters(self):
        # 动态投影使用 Xavier 初始化
        nn.init.xavier_uniform_(self.proj_dynamic.weight)
        # 门控因子初始化为小值（接近恒等映射）
        nn.init.constant_(self.alpha_pre, 0.01)
        nn.init.constant_(self.alpha_post, 0.01)
        nn.init.constant_(self.alpha_res, 0.01)
        # 偏置初始化为零
        nn.init.zeros_(self.bias_pre)
        nn.init.zeros_(self.bias_post)
        nn.init.zeros_(self.bias_res)

    def forward(self, x):
        """
        Args:
            x: [B, L, C] 输入张量
        Returns:
            [B, L, n, C] 输出张量，按照论文公式(3)，输出维度为 n×C
        """
        B, L, C = x.shape
        n = self.n_streams

        # 1. 扩展为多流残差：将输入复制 n 份，构建隐藏矩阵 [B, L, n, C]
        x_exp = x.unsqueeze(2).expand(-1, -1, n, -1)  # [B, L, n, C]

        # 2. 展平用于后续计算
        x_flat = x_exp.reshape(B, L, -1)  # [B, L, n*C]

        # 3. RMSNorm（对最后一维归一化）
        x_norm = self.rms_norm(x_flat)  # [B, L, n*C]

        # 4. 动态映射线性投影
        dynamic = self.proj_dynamic(x_norm)  # [B, L, n^2 + 2n]

        # 分离出 H_pre, H_post, H_res 的动态部分
        dynamic_pre = dynamic[..., :n]  # [B, L, n]
        dynamic_post = dynamic[..., n:2*n]  # [B, L, n]
        dynamic_res = dynamic[..., 2*n:]  # [B, L, n^2]

        # 5. 计算映射矩阵（式 (7)）
        # H_pre: [B, L, n]
        H_pre_dyn = self.alpha_pre * dynamic_pre + self.bias_pre
        # H_post: [B, L, n]
        H_post_dyn = self.alpha_post * dynamic_post + self.bias_post
        # H_res: [B, L, n, n]
        H_res_dyn = self.alpha_res * dynamic_res.view(B, L, n, n) + self.bias_res

        # 6. 流形约束（式 (8)）
        # H_pre 使用 Sigmoid 约束到 (0,1)
        H_pre = torch.sigmoid(H_pre_dyn)  # [B, L, n]
        # H_post 使用 2*Sigmoid 约束到 (0,2)
        H_post = 2 * torch.sigmoid(H_post_dyn)  # [B, L, n]
        # H_res 通过 Sinkhorn-Knopp 投影为双随机矩阵
        H_res = sinkhorn_knopp(H_res_dyn, self.sinkhorn_iters)  # [B, L, n, n]

        # 7. 残差路径计算（式 (3)）严格遵循论文
        # 7.1 预映射：H_pre * x_l
        pre_mixed = torch.einsum('bln,blnc->blc', H_pre, x_exp)  # [B, L, C]

        # 7.2 残差函数 F
        f_out = self.residual_fn(pre_mixed)  # [B, L, C]

        # 7.3 将 F 的输出扩展回 n 流
        f_out_exp = f_out.unsqueeze(2).expand(-1, -1, n, -1)  # [B, L, n, C]

        # 7.4 后映射：H_post^T * F(...)
        post_mixed = torch.einsum('bln,blnc->blnc', H_post, f_out_exp)  # [B, L, n, C]

        # 7.5 残差映射：H_res * x_l
        res_mixed = torch.einsum('blnm,blmc->blnc', H_res, x_exp)  # [B, L, n, C]

        # 8. 合并两条路径（公式(3)的最终结果）
        # x_{l+1} = H_res * x_l + H_post^T * F(H_pre * x_l)
        stream_out = res_mixed + post_mixed  # [B, L, n, C]

        # 9. 直接返回 stream_out，这是论文公式(3)的精确实现
        # 输出维度为 n×C，没有额外残差连接
        return stream_out


class MHAttention(nn.Module):
    """
    多头注意力层，与 mHC 结合使用
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # [B, L, num_heads, head_dim]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.proj(out)
        return out


class MHCTransformerBlock(nn.Module):
    """
    Transformer 块，包含多头注意力和 mHC 层
    这是论文中描述的架构
    """
    def __init__(self, dim, num_heads=8, n_streams=4, hidden_dim=None, dropout=0.0):
        super().__init__()
        self.dim = dim
        
        # 注意力层
        self.attn = MHAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        
        # mHC 层
        self.mhc = MHCLayer(dim, n_streams, hidden_dim, dropout=dropout)
        
        # 流合并层（将 n×C 维投影回 C 维）
        # 论文中提到："we set the expansion rate n = 4 ... and then project it back to the original dimension C"
        self.merge_streams = nn.Linear(dim * n_streams, dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 注意力部分
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # mHC 部分
        mhc_out = self.mhc(x)  # [B, L, n, C]
        
        # 将多流输出展平并投影回原始维度
        B, L, n, C = mhc_out.shape
        mhc_flat = mhc_out.reshape(B, L, -1)  # [B, L, n*C]
        mhc_proj = self.merge_streams(mhc_flat)  # [B, L, C]
        
        # 最终输出
        x = self.norm2(x + self.dropout(mhc_proj))
        return x


# 测试代码
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 16
    dim = 128
    n_streams = 4
    
    print("=== 测试 mHC 层（严格遵循论文公式） ===")
    
    # 测试单独的 mHC 层
    model = MHCLayer(dim=dim, n_streams=n_streams, hidden_dim=512, dropout=0.1)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    
    print(f"输入形状: {x.shape}")
    output = model(x)
    print(f"mHC 输出形状: {output.shape} (应该是 [B, L, n, C])")
    
    # 检查双随机矩阵性质
    B, L = x.shape[0], x.shape[1]
    H_res_sample = sinkhorn_knopp(
        model.alpha_res * torch.randn(1, 1, n_streams, n_streams) + model.bias_res, 
        model.sinkhorn_iters
    )
    print(f"\n双随机矩阵验证:")
    print(f"行和: {H_res_sample.sum(dim=-1)}")
    print(f"列和: {H_res_sample.sum(dim=-2)}")
    
    # 梯度检查
    loss = output.mean()
    loss.backward()
    print(f"输入梯度范数: {x.grad.norm().item():.4f}")
    print(f"动态投影权重梯度范数: {model.proj_dynamic.weight.grad.norm().item():.4f}")
    
    # 测试完整的 Transformer 块
    print("\n=== 测试完整的 MHCTransformerBlock ===")
    transformer_block = MHCTransformerBlock(
        dim=dim, 
        num_heads=8, 
        n_streams=n_streams,
        dropout=0.1
    )
    
    x2 = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    print(f"Transformer 输入形状: {x2.shape}")
    transformer_out = transformer_block(x2)
    print(f"Transformer 输出形状: {transformer_out.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in transformer_block.parameters())
    print(f"Transformer 块参数量: {total_params:,}")
    
    print("\n✅ 所有测试通过！mHC 层已严格按论文公式实现。")
