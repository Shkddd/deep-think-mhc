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
        norm = x.norm(2, dim=-1, keepdim=True)
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

        # 输出投影（将 n*C 维映射回 C 维）
        self.out_proj = nn.Linear(dim * n_streams, dim)

        self._init_parameters()

    def _init_parameters(self):
        # 动态投影使用 Xavier 初始化
        nn.init.xavier_uniform_(self.proj_dynamic.weight)
        # 门控因子初始化为小值
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
            [B, L, C] 输出张量
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

        # 7. 残差路径计算（式 (3)）
        # 7.1 预映射：H_pre * x_l
        pre_mixed = torch.einsum('bln,blnc->blc', H_pre, x_exp)  # [B, L, C]

        # 7.2 残差函数 F
        f_out = self.residual_fn(pre_mixed)  # [B, L, C]

        # 7.3 将 F 的输出扩展回 n 流
        f_out_exp = f_out.unsqueeze(2).expand(-1, -1, n, -1)  # [B, L, n, C]

        # 7.4 后映射：H_post^T * F(...)
        # 注意：论文中 H_post 是行向量，这里使用转置进行乘法
        post_mixed = torch.einsum('bln,blnc->blnc', H_post, f_out_exp)  # [B, L, n, C]

        # 7.5 残差映射：H_res * x_l
        res_mixed = torch.einsum('blnm,blmc->blnc', H_res, x_exp)  # [B, L, n, C]

        # 8. 合并两条路径
        stream_out = res_mixed + post_mixed  # [B, L, n, C]

        # 9. 将多流输出合并为单流
        merged = stream_out.reshape(B, L, -1)  # [B, L, n*C]
        output = self.out_proj(merged)  # [B, L, C]

        # 10. 残差连接（保持恒等映射）
        output = output + x

        return output

# 测试代码
if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 16
    dim = 128
    n_streams = 4

    model = MHCLayer(dim=dim, n_streams=n_streams, hidden_dim=512, dropout=0.1)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")

    # 梯度检查
    loss = output.mean()
    loss.backward()
    print(f"Input grad norm: {x.grad.norm().item():.4f}")
    print(f"Dynamic projection weight grad norm: {model.proj_dynamic.weight.grad.norm().item():.4f}")
    
    # 打印一些额外的信息
    print(f"\nModel parameters:")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"alpha_pre value: {model.alpha_pre.item():.6f}")
    print(f"alpha_post value: {model.alpha_post.item():.6f}")
    print(f"alpha_res value: {model.alpha_res.item():.6f}")
    
    # 检查双随机矩阵的性质
    B, L = x.shape[0], x.shape[1]
    H_res_sample = sinkhorn_knopp(model.alpha_res * torch.randn(1, 1, n_streams, n_streams) + model.bias_res, model.sinkhorn_iters)
    print(f"\nSample H_res properties:")
    print(f"H_res shape: {H_res_sample.shape}")
    print(f"Row sums: {H_res_sample.sum(dim=-1)}")
    print(f"Column sums: {H_res_sample.sum(dim=-2)}")
    
    print("\n前向与反向传播测试通过。")
