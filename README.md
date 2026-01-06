# deep-think-mhc

**Deep Think on mHC: A Faithful PyTorch Reimplementation of DeepSeek's Manifold-Constrained Hyper-Connections**

This repository explores and reimplements the groundbreaking **mHC (Manifold-Constrained Hyper-Connections)** architecture from DeepSeek-AI's recent paper:

**mHC: Manifold-Constrained Hyper-Connections**  
Zhenda Xie et al., DeepSeek-AI  
[arXiv:2512.24880](https://arxiv.org/abs/2512.24880) (December 2025)

The implementation integrates the full mHC mechanism into a clean, minimal Transformer model based on Andrej Karpathy's **nanoGPT**, replacing **both** standard residual connections (post-attention and post-FFN) with the exact formulation described in the paper.

### Why This Project?

Standard residual connections have been the cornerstone of deep networks since ResNet (2015). Hyper-Connections (HC) offered greater expressivity by introducing learnable mixing across expanded streams, but suffered from severe training instability at depth and width.

**mHC** elegantly solves this by projecting the connection matrices onto a doubly-stochastic manifold using Sinkhorn-Knopp normalization, preserving signal conservation while enabling richer, learnable residual pathways.

This repo provides a **highly accurate, readable reproduction** of mHC for research, experimentation, and deeper understanding — perfect for "deep thinking" about modern Transformer improvements.

### Key Features

- **Pixel-perfect fidelity** to the paper's mHC formulation:
  - Multi-stream expansion (`n=4` by default)
  - Flattened RMSNorm across streams
  - Learnable gating scalars `α_pre`, `α_post`, `α_res` (initialized to 0.01)
  - Bias-initialized projections for near-identity start (large positive/negative biases)
  - Sigmoid gating for `H_pre`, 2×sigmoid for `H_post`
  - Sinkhorn-Knopp (20 iterations) for doubly-stochastic `H_res`
  - Exact matrix multiplications matching Equations (4)–(8) in the paper
- Drop-in replacement for residual connections in any Transformer block
- Built on nanoGPT-style architecture for clarity and minimalism
- No external dependencies beyond PyTorch
- Small demo model for instant testing

### Installation

```bash
git clone https://github.com/yourusername/deep-think-mhc.git
cd deep-think-mhc
pip install torch  # PyTorch 2.0+ recommended
```

### Quick Start

Run the demo script:

```bash
python mhc_nanogpt.py
```

This creates a tiny GPT model (2 layers, 64-dim embeddings) with mHC residuals and runs a forward pass on random tokens.

### Usage in Your Own Models

```python
from mhc_layer import mHCLayer
from block import Block  # Already uses mHCLayer for both residuals

# Or manually:
def forward(self, x):
    x = self.mhc_attn(self.ln_1(x), self.attn)
    x = self.mhc_ffn(self.ln_2(x), self.mlp)
    return x
```

### Configuration Options (in `mHCLayer`)

| Parameter         | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `dim`             | -       | Hidden dimension                         |
| `expansion_rate`  | 4       | Number of streams `n` (as in paper)      |
| `sinkhorn_iter`   | 20      | Sinkhorn-Knopp iterations (paper uses 20)|
| `gate_init`       | 0.01    | Initial value for α parameters           |

### Citation

If you use this implementation or find it helpful, please cite the original DeepSeek paper:

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Zhenda Xie and others},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```

### Notes

- This is an **independent, research-oriented implementation** — not affiliated with DeepSeek-AI.
- Sinkhorn iterations add minor overhead (~7% as reported in the paper); for large-scale training, consider optimized kernels.
- Contributions, bug reports, and experiments are very welcome!

Dive deep into the future of Transformer residuals. 🚀
