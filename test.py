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

def sinkhorn_knopp(log_mat, num_iters=3, eps=1e-8):
    """
    简化的 Sinkhorn-Knopp 迭代，只做 3 次迭代减少计算量
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
    def __init__(self, dim=4, n_streams=2, hidden_dim=8, sinkhorn_iters=3, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

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
        self.alpha_pre = nn.Parameter(torch.tensor([0.01]))
        self.alpha_post = nn.Parameter(torch.tensor([0.01]))
        self.alpha_res = nn.Parameter(torch.tensor([0.01]))

        # 残差函数 F（简化MLP）
        self.residual_fn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

        # 初始化权重为小值以便观察
        with torch.no_grad():
            nn.init.normal_(self.proj_dynamic.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.residual_fn[0].weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.residual_fn[0].bias)
            nn.init.normal_(self.residual_fn[2].weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.residual_fn[2].bias)

    def forward(self, x):
        print(f"\n{'='*60}")
        print("开始前向传播计算")
        print(f"{'='*60}")
        B, L, C = x.shape
        n = self.n_streams
        
        print(f"1. 输入形状: x.shape = {x.shape}")
        print(f"   batch_size = {B}, seq_len = {L}, dim = {C}, n_streams = {n}")
        print(f"   输入值示例: x[0,0,:] = {x[0,0,:]}")
        
        # 1. 扩展为多流残差
        x_exp = x.unsqueeze(2).expand(-1, -1, n, -1)
        print(f"\n2. 多流扩展后: x_exp.shape = {x_exp.shape}")
        print(f"   扩展后示例: x_exp[0,0,0,:] = {x_exp[0,0,0,:]}")
        
        # 2. 展平用于后续计算
        x_flat = x_exp.reshape(B, L, -1)
        print(f"\n3. 展平后: x_flat.shape = {x_flat.shape}")
        print(f"   展平后示例: x_flat[0,0,:] = {x_flat[0,0,:]}")
        
        # 3. RMSNorm
        x_norm = self.rms_norm(x_flat)
        print(f"\n4. RMSNorm后: x_norm.shape = {x_norm.shape}")
        print(f"   RMSNorm weight形状: {self.rms_norm.weight.shape}")
        print(f"   归一化后示例: x_norm[0,0,:] = {x_norm[0,0,:]}")
        
        # 4. 动态映射线性投影
        dynamic = self.proj_dynamic(x_norm)
        print(f"\n5. 动态投影后: dynamic.shape = {dynamic.shape}")
        print(f"   投影权重形状: proj_dynamic.weight.shape = {self.proj_dynamic.weight.shape}")
        print(f"   动态值示例: dynamic[0,0,:] = {dynamic[0,0,:]}")
        
        # 分离出 H_pre, H_post, H_res 的动态部分
        dynamic_pre = dynamic[..., :n]
        dynamic_post = dynamic[..., n:2*n]
        dynamic_res = dynamic[..., 2*n:]
        print(f"\n6. 动态部分分割:")
        print(f"   dynamic_pre.shape = {dynamic_pre.shape}, 示例: {dynamic_pre[0,0,:]}")
        print(f"   dynamic_post.shape = {dynamic_post.shape}, 示例: {dynamic_post[0,0,:]}")
        print(f"   dynamic_res.shape = {dynamic_res.shape}, 展开后形状: [{B}, {L}, {n}, {n}]")
        
        # 5. 计算映射矩阵
        H_pre_dyn = self.alpha_pre * dynamic_pre + self.bias_pre
        H_post_dyn = self.alpha_post * dynamic_post + self.bias_post
        H_res_dyn = self.alpha_res * dynamic_res.view(B, L, n, n) + self.bias_res
        print(f"\n7. 动态+静态映射矩阵计算:")
        print(f"   α_pre = {self.alpha_pre.item():.4f}, α_post = {self.alpha_post.item():.4f}, α_res = {self.alpha_res.item():.4f}")
        print(f"   bias_pre形状: {self.bias_pre.shape}, 值: {self.bias_pre}")
        print(f"   H_pre_dyn形状: {H_pre_dyn.shape}, 示例: {H_pre_dyn[0,0,:]}")
        print(f"   H_post_dyn形状: {H_post_dyn.shape}, 示例: {H_post_dyn[0,0,:]}")
        print(f"   H_res_dyn形状: {H_res_dyn.shape}, 示例[0,0,:,:]:\n{H_res_dyn[0,0,:,:]}")
        
        # 6. 流形约束
        H_pre = torch.sigmoid(H_pre_dyn)
        H_post = 2 * torch.sigmoid(H_post_dyn)
        H_res = sinkhorn_knopp(H_res_dyn, self.sinkhorn_iters)
        print(f"\n8. 流形约束后:")
        print(f"   H_pre = σ(H_pre_dyn) 形状: {H_pre.shape}, 示例: {H_pre[0,0,:]}")
        print(f"   H_post = 2σ(H_post_dyn) 形状: {H_post.shape}, 示例: {H_post[0,0,:]}")
        print(f"   H_res = Sinkhorn(H_res_dyn) 形状: {H_res.shape}")
        print(f"   H_res示例[0,0,:,:]:\n{H_res[0,0,:,:]}")
        print(f"   验证双随机性 - 行和: {H_res[0,0,:,:].sum(dim=-1)}")
        print(f"                    列和: {H_res[0,0,:,:].sum(dim=-2)}")
        
        # 7. 残差路径计算
        # 7.1 预映射：H_pre * x_l
        pre_mixed = torch.einsum('bln,blnc->blc', H_pre, x_exp)
        print(f"\n9. 预映射计算:")
        print(f"   pre_mixed = einsum('bln,blnc->blc', H_pre, x_exp)")
        print(f"   pre_mixed形状: {pre_mixed.shape}")
        print(f"   示例: pre_mixed[0,0,:] = {pre_mixed[0,0,:]}")
        
        # 7.2 残差函数 F
        f_out = self.residual_fn(pre_mixed)
        print(f"\n10. 残差函数F:")
        print(f"   残差函数结构: Linear({C}→{self.residual_fn[0].out_features}) → GELU → Linear(→{C})")
        print(f"   f_out形状: {f_out.shape}")
        print(f"   示例: f_out[0,0,:] = {f_out[0,0,:]}")
        
        # 7.3 将 F 的输出扩展回 n 流
        f_out_exp = f_out.unsqueeze(2).expand(-1, -1, n, -1)
        print(f"\n11. 扩展F输出到多流:")
        print(f"   f_out_exp形状: {f_out_exp.shape}")
        
        # 7.4 后映射：H_post^T * F(...)
        post_mixed = torch.einsum('bln,blnc->blnc', H_post, f_out_exp)
        print(f"\n12. 后映射计算:")
        print(f"   post_mixed = einsum('bln,blnc->blnc', H_post, f_out_exp)")
        print(f"   post_mixed形状: {post_mixed.shape}")
        print(f"   示例[0,0,0,:] = {post_mixed[0,0,0,:]}")
        
        # 7.5 残差映射：H_res * x_l
        res_mixed = torch.einsum('blnm,blmc->blnc', H_res, x_exp)
        print(f"\n13. 残差映射计算:")
        print(f"   res_mixed = einsum('blnm,blmc->blnc', H_res, x_exp)")
        print(f"   res_mixed形状: {res_mixed.shape}")
        print(f"   示例[0,0,0,:] = {res_mixed[0,0,0,:]}")
        
        # 8. 合并两条路径
        stream_out = res_mixed + post_mixed
        print(f"\n14. 最终输出计算:")
        print(f"   stream_out = res_mixed + post_mixed")
        print(f"   stream_out形状: {stream_out.shape}")
        print(f"   示例[0,0,0,:] = {stream_out[0,0,0,:]}")
        
        return stream_out

# 设置小参数以减少计算量
print("="*70)
print("mHC 层详细计算演示")
print("参数配置: batch=1, seq_len=2, dim=4, n_streams=2, hidden_dim=8")
print("="*70)

torch.manual_seed(42)

# 创建模型
model = MHCLayer(dim=4, n_streams=2, hidden_dim=8, sinkhorn_iters=3)

# 创建很小的输入
x = torch.randn(1, 2, 4, requires_grad=True)
print(f"\n初始化输入 x:")
print(f"形状: {x.shape}")
print(f"值:\n{x}")

# 前向传播
print("\n" + "="*70)
print("开始前向传播...")
print("="*70)

output = model(x)

print(f"\n" + "="*70)
print("前向传播完成！")
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"输出值示例:\n{output[0,0,:,:]}")

# 计算损失（简单使用均值）
print("\n" + "="*70)
print("计算损失...")
print("="*70)
loss = output.mean()
print(f"损失值: loss = output.mean() = {loss.item():.6f}")

# 反向传播
print("\n" + "="*70)
print("开始反向传播...")
print("="*70)

loss.backward()

print(f"\n反向传播完成！")
print(f"梯度检查:")

# 检查梯度是否存在
gradients_exist = True
print(f"1. 输入梯度:")
print(f"   x.grad形状: {x.grad.shape}")
print(f"   x.grad范数: {x.grad.norm().item():.6f}")
print(f"   x.grad示例: {x.grad[0,0,:]}")

print(f"\n2. 动态投影权重梯度:")
print(f"   proj_dynamic.weight.grad形状: {model.proj_dynamic.weight.grad.shape}")
print(f"   proj_dynamic.weight.grad范数: {model.proj_dynamic.weight.grad.norm().item():.6f}")

print(f"\n3. α参数梯度:")
print(f"   α_pre梯度: {model.alpha_pre.grad.item():.6f}")
print(f"   α_post梯度: {model.alpha_post.grad.item():.6f}")
print(f"   α_res梯度: {model.alpha_res.grad.item():.6f}")

print(f"\n4. 偏置梯度:")
print(f"   bias_pre梯度形状: {model.bias_pre.grad.shape}")
print(f"   bias_pre梯度值: {model.bias_pre.grad}")
print(f"   bias_post梯度形状: {model.bias_post.grad.shape}")
print(f"   bias_post梯度值: {model.bias_post.grad}")
print(f"   bias_res梯度形状: {model.bias_res.grad.shape}")
print(f"   bias_res梯度值:\n{model.bias_res.grad}")

print(f"\n5. 残差函数梯度:")
print(f"   第一层权重梯度形状: {model.residual_fn[0].weight.grad.shape}")
print(f"   第一层权重梯度范数: {model.residual_fn[0].weight.grad.norm().item():.6f}")
print(f"   第二层权重梯度形状: {model.residual_fn[2].weight.grad.shape}")
print(f"   第二层权重梯度范数: {model.residual_fn[2].weight.grad.norm().item():.6f}")

# 计算 FLOPs 估算
print("\n" + "="*70)
print("计算量分析 (近似FLOPs)")
print("="*70)

B, L, C = x.shape
n = model.n_streams
hidden_dim = model.residual_fn[0].out_features

# 主要计算操作估算
flops_info = {
    "1. 多流扩展": B * L * n * C * 1,  # 复制操作
    "2. 展平操作": B * L * n * C * 1,
    "3. RMSNorm": B * L * n * C * 5,  # 平方、均值、开方、除法、乘法
    "4. 动态投影": B * L * (n*C) * (n*n + 2*n) * 2,  # 矩阵乘法
    "5. Sinkhorn迭代": B * L * n * n * 3 * 6,  # 3次迭代 * (exp+除法*2) * 行列归一化
    "6. Sigmoid约束": B * L * (n + n) * 10,  # 两个Sigmoid
    "7. 预映射 (einsum)": B * L * n * C * 2,  # H_pre * x_exp
    "8. 残差函数MLP": B * L * (C * hidden_dim * 2 + hidden_dim * C * 2),  # 两个线性层
    "9. 后映射 (einsum)": B * L * n * C * 2,  # H_post * f_out_exp
    "10. 残差映射 (einsum)": B * L * n * n * C * 2,  # H_res * x_exp
    "11. 加法操作": B * L * n * C * 1,
}

total_flops = sum(flops_info.values())
print("各操作FLOPs估算:")
for key, value in flops_info.items():
    percentage = (value / total_flops) * 100
    print(f"  {key:30} {value:10,} FLOPs ({percentage:5.1f}%)")

print(f"\n总FLOPs估算: {total_flops:,}")

# 内存占用估算
print("\n" + "="*70)
print("内存占用分析")
print("="*70)

# 主要张量大小估算
memory_info = {
    "输入 x": B * L * C * 4,  # float32 = 4 bytes
    "多流扩展 x_exp": B * L * n * C * 4,
    "展平 x_flat": B * L * n * C * 4,
    "归一化 x_norm": B * L * n * C * 4,
    "动态投影输出": B * L * (n*n + 2*n) * 4,
    "映射矩阵 (3个)": B * L * (n + n + n*n) * 4,
    "预映射结果": B * L * C * 4,
    "残差函数中间": B * L * hidden_dim * 4,
    "残差函数输出": B * L * C * 4,
    "后映射结果": B * L * n * C * 4,
    "残差映射结果": B * L * n * C * 4,
    "最终输出": B * L * n * C * 4,
}

# 计算峰值内存（假设所有张量同时存在）
peak_memory = sum(memory_info.values()) / 1024  # 转换为KB

print("各张量内存占用:")
for key, value in memory_info.items():
    size_kb = value / 1024
    print(f"  {key:25} {size_kb:10.2f} KB")

print(f"\n峰值内存估算: {peak_memory:.2f} KB ({peak_memory/1024:.2f} MB)")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,} 个参数")
print(f"参数内存: {sum(p.numel() for p in model.parameters()) * 4 / 1024:.2f} KB")

print("\n" + "="*70)
print("计算完成！所有前向传播和反向传播步骤正常。")
print("="*70)
