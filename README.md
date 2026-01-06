Manifold-Constrained Hyper-Connections (mHC)

PyTorch implementation of the Manifold-Constrained Hyper-Connections (mHC) layer from DeepSeek's research paper.

📄 Paper Reference

· Title: mHC: Manifold-Constrained Hyper-Connections
· Authors: DeepSeek (深度求索)
· arXiv: https://arxiv.org/abs/2512.24880
· PDF: https://arxiv.org/pdf/2512.24880.pdf
· Hugging Face: https://huggingface.co/papers/2512.24880

🚀 Overview

mHC introduces a novel neural network layer that enhances model expressivity through multi-stream competition and manifold constraints. The core innovation is the use of doubly-stochastic matrices (via Sinkhorn-Knopp algorithm) to model competitive relationships between parallel processing streams, enabling more efficient information flow while maintaining gradient stability.

✨ Key Features

· Multi-Stream Competition: Parallel processing streams with competitive interactions
· Doubly-Stochastic Constraints: Enforced via Sinkhorn-Knopp normalization
· Manifold-Constrained Mapping: Dynamic + static mapping with manifold constraints
· Residual Learning: Preserves identity mapping for training stability
· Gradient-Friendly: All operations maintain gradient flow

📦 Installation

```bash
pip install torch>=1.9.0
```

🎯 Quick Start

```python
import torch
from mhc_layer import MHCLayer

# Initialize mHC layer
model = MHCLayer(
    dim=128,               # Input dimension
    n_streams=4,           # Number of parallel streams
    hidden_dim=512,        # Hidden dimension for residual function
    sinkhorn_iters=20,     # Sinkhorn iterations
    dropout=0.1            # Dropout probability
)

# Forward pass
x = torch.randn(4, 16, 128)  # [batch, seq_len, dim]
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

🏗️ Architecture

Core Formula (from paper)

The mHC layer follows this formulation:

```
x_{l+1} = H_res^l x_l + (H_post^l)^T F(H_pre^l x_l, W_l)
```

With manifold constraints:

```
H_pre = σ(H̃_pre)                # Constrained to (0,1)
H_post = 2σ(H̃_post)             # Constrained to (0,2)
H_res = Sinkhorn-Knopp(H̃_res)   # Doubly-stochastic matrix
```

Components

1. Multi-Stream Expansion: Input features expanded to n parallel streams
2. Dynamic-Static Mapping: Hybrid mapping matrices combining input-dependent and learnable components
3. Manifold Constraints:
   · Sigmoid for H_pre (0,1)
   · 2×Sigmoid for H_post (0,2)
   · Sinkhorn-Knopp for H_res (doubly-stochastic)
4. Residual Function F: Non-linear transformation (MLP with GELU activation)
5. Stream Fusion: Competitive aggregation back to single stream

📊 Performance Characteristics

Parameter Value Description
Parameters ~1.7M For dim=128, n_streams=4
Memory Efficient Uses expand() instead of repeat()
Gradients Stable RMSNorm + proper initialization
Numerical Stability High eps protection in all divisions

🧪 Testing

The implementation includes comprehensive tests:

```bash
python mhc_layer.py
```

Expected output:

```
Input shape: torch.Size([4, 16, 128])
Output shape: torch.Size([4, 16, 128])
Input grad norm: 0.0285
Dynamic projection weight grad norm: 0.0437
Model parameters: 1,704,580
Sample H_res properties: Row/Column sums = 1.0
```

🔧 Configuration Options

```python
MHCLayer(
    dim=128,                    # Input feature dimension (C)
    n_streams=4,                # Expansion rate (n), typically 4-8
    hidden_dim=512,             # Hidden dim for residual MLP (default: dim*4)
    sinkhorn_iters=20,          # Sinkhorn-Knopp iterations (3-30)
    dropout=0.1,                # Dropout rate for regularization
)
```

🎯 Use Cases

· Transformer Enhancements: Replace FFN layers with mHC blocks
· Vision Transformers: Competitive feature processing
· Graph Neural Networks: Multi-stream node/edge processing
· Recurrent Architectures: Competitive memory mechanisms

📈 Training Tips

1. Initialization: α factors initialized to 0.01 for identity mapping
2. Learning Rate: Similar to standard Transformer layers
3. Batch Size: Works well with standard batch sizes (16-128)
4. Regularization: Built-in dropout in residual function

🧠 Theoretical Insights

mHC introduces several novel concepts:

1. Hyper-Connections: Connections between parallel processing streams
2. Manifold Constraints: Mathematical guarantees on mapping matrices
3. Competitive Learning: Streams compete and cooperate through doubly-stochastic matrices
4. Dynamic Adaptation: Input-dependent modulation of connection strengths

🔬 Advanced Usage

Multi-Scale mHC

```python
class MultiScaleMHC(nn.Module):
    def __init__(self, dim, n_streams=4, num_scales=3):
        super().__init__()
        self.layers = nn.ModuleList([
            MHCLayer(dim, max(2, n_streams // (2**i)))
            for i in range(num_scales)
        ])
```

Integration with Transformers

```python
class TransformerWithMHC(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(dim, num_heads),
                'mhc': MHCLayer(dim, n_streams=4),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim),
            }) for _ in range(num_layers)
        ])
```

📚 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@misc{deepseek2024mhc,
      title={mHC: Manifold-Constrained Hyper-Connections}, 
      author={DeepSeek},
      year={2024},
      eprint={2512.24880},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

🛠️ Implementation Notes

· RMSNorm: Custom implementation included (not in standard PyTorch)
· Sinkhorn-Knopp: Differentiable implementation with gradient support
· Einsum Operations: Used for clarity and efficiency
· Memory Efficient: Careful tensor operations to avoid unnecessary copies

📱 Compatibility

· PyTorch: ≥ 1.9.0
· Python: ≥ 3.7
· CUDA: Optional, supports GPU acceleration
· Platform: Linux, macOS, Windows

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

📄 License

This implementation is provided under the MIT License. See LICENSE file for details.

📧 Contact

For questions about the implementation, please open an issue on GitHub.

---

Note: This is a research implementation. For production use, additional optimizations and testing may be required.
