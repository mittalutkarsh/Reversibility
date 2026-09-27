"""
scripts/model.py

One TransformerBlock, three ways of stacking it.

The block computes f(x) -- the *update*, with its layer norms on the inside.
Keeping the norms inside f is what lets the identical block serve all three
stack rules, because each rule only decides how f's output is combined:

    baseline   x       = x      + f(x)          not reversible
    euler      p[l+1]  = p[l]   + h  * f(p[l])  not exactly reversible (see below)
    midpoint   p[l+1]  = p[l-1] + 2h * f(p[l])  reversible

With f(x) = attn(ln1(x)) + mlp(ln2(x + attn(ln1(x)))), the baseline rule
x + f(x) is *algebraically identical* to the ordinary pre-norm transformer
block written as two residual steps:

    x1 = x  + attn(ln1(x))
    x2 = x1 + mlp(ln2(x1))        ==   x + f(x)

so "norm inside f" is a refactor of the standard block, not a different model.

REVERSIBILITY, stated honestly
------------------------------
midpoint  is exactly invertible.  p[l-1] = p[l+1] - 2h*f(p[l]) evaluates f at
          p[l], a state the backward walk already holds.  You must carry a
          *pair* of adjacent states, not one.  Exact up to float rounding.

euler     is NOT exactly invertible.  Solving p[l+1] = p[l] + h*f(p[l]) for
          p[l] gives p[l] = p[l+1] - h*f(p[l]) -- f is evaluated at the
          unknown.  That is an implicit equation with no closed form; an exact
          inverse would need a fixed-point or Newton solve per layer, which is
          an iterative approximation, not an inverse.  check_reversibility.py
          therefore uses the naive explicit inverse
              p_hat[l] = p[l+1] - h*f(p[l+1])
          and reports the resulting error as a FAIL.  This is a real property
          of the forward Euler map, not a bug and not a tuning problem.

PRECISION
---------
fp32 only for the reversibility gate.  bf16 autocast trains fine on this
machine (see environment_check.py) but its 8-bit mantissa destroys the
reconstruction, which subtracts two nearby numbers.  check_reversibility.py
runs both so the gap is on the record.

DROPOUT
-------
Exactly 0, enforced by construction: no nn.Dropout module is created anywhere,
so the forward pass draws no random numbers at all.  That is what makes
recomputing a block reproduce its forward activations bit for bit.  A nonzero
dropout mask would be resampled on recomputation and the reconstruction would
silently diverge.

POSITIONS
---------
RoPE, which has no parameters.  A learned 1024x512 position table would add
524,288 params and push the total to 20.48M; parameter-free positions land at
19,957,248 ~= the 19.9M in the spec.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reversible import midpoint_reversible

SEED = 1337

# ---------------------------------------------------------------- config
VOCAB_SIZE = 8192
D_MODEL = 512
N_LAYERS = 5
N_HEADS = 8
D_FF = 4 * D_MODEL          # 2048
CONTEXT = 1024
DROPOUT = 0.0

H = 0.25                    # integrator step size -- do not tune
BLEND = 0.5                 # midpoint readout blend -- do not tune

PARAM_MIN = 19_500_000
PARAM_MAX = 20_500_000

VARIANTS = ("baseline", "euler", "midpoint", "midpoint_rev")


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    d_ff: int = D_FF
    context: int = CONTEXT
    dropout: float = DROPOUT
    variant: str = "baseline"
    h: float = H
    blend: float = BLEND

    def __post_init__(self):
        assert self.variant in VARIANTS, f"unknown variant {self.variant!r}"
        assert self.dropout == 0.0, "dropout must be exactly 0 for reversibility"
        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads


# ------------------------------------------------------------------ rope
def build_rope_cache(seq_len, head_dim, device, dtype):
    """cos/sin tables of shape (seq_len, head_dim//2)."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, n_heads, T, head_dim). Rotates even/odd channel pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


# ------------------------------------------------------------- submodules
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        # No nn.Dropout: dropout is 0 by construction, see module docstring.

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class TransformerBlock(nn.Module):
    """f(x): the block update. Used unchanged by all three stack variants.

    Returns only the update, never x + update -- the stacking rule owns the
    combination. Deterministic given x: no dropout, no RNG, so a recomputation
    during the backward walk reproduces the forward value bit for bit.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        a = self.attn(self.ln1(x), cos, sin)
        m = self.mlp(self.ln2(x + a))
        return a + m


# -------------------------------------------------------------------- gpt
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, verbose: bool = True):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        self.apply(self._init_weights)
        self._rope = {}

        n = self.num_params()
        if verbose:
            print(f"[model] variant={cfg.variant:<9} params={n:,} ({n / 1e6:.2f}M)")
        assert PARAM_MIN <= n <= PARAM_MAX, (
            f"parameter count {n:,} outside [{PARAM_MIN:,}, {PARAM_MAX:,}] -- "
            f"stop and report, do not adjust the config"
        )

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # .parameters() de-duplicates the tied embedding/head weight.
        return sum(p.numel() for p in self.parameters())

    def rope(self, T, device, dtype):
        key = (T, str(device), dtype)
        if key not in self._rope:
            self._rope[key] = build_rope_cache(T, self.cfg.head_dim, device, dtype)
        return self._rope[key]

    def embed(self, idx):
        return self.tok_emb(idx)

    # ---- the three stacking rules -----------------------------------
    def run_stack(self, p0, cos, sin):
        """Returns (final_hidden, states) where states[i] is p[i]."""
        cfg = self.cfg
        h, blend = cfg.h, cfg.blend
        states = [p0]

        if cfg.variant == "baseline":
            x = p0
            for blk in self.blocks:
                x = x + blk(x, cos, sin)
                states.append(x)
            return states[-1], states

        if cfg.variant == "euler":
            x = p0
            for blk in self.blocks:
                x = x + h * blk(x, cos, sin)
                states.append(x)
            return states[-1], states

        if cfg.variant == "midpoint_rev":
            # Same arithmetic as "midpoint", but the backward reconstructs
            # activations instead of storing them. Returns no per-layer states
            # -- not keeping them is the entire point. See reversible.py.
            final = midpoint_reversible(p0, cos, sin, self.blocks, h, blend)
            return final, None

        # midpoint / leapfrog.
        #
        # Layer 0 needs p[-1], which does not exist. It is run as an ordinary
        # residual block (p[1] = p[0] + f0(p[0]), the baseline rule) purely to
        # manufacture the second state the two-term recurrence needs. This
        # bootstrap step is never inverted: the backward walk recovers p[0]
        # from layer 1's inverse (p[0] = p[2] - 2h*f1(p[1])), so the choice of
        # bootstrap does not affect reversibility at all -- only the function
        # the stack computes. The literal reading of the spec ("an ordinary
        # residual block") is used rather than a scaled Euler step p[0] +
        # h*f0(p[0]), which would keep step sizes uniform.
        p_prev = p0
        p_cur = p0 + self.blocks[0](p0, cos, sin)
        states.append(p_cur)

        for blk in self.blocks[1:]:
            p_next = p_prev + 2.0 * h * blk(p_cur, cos, sin)
            states.append(p_next)
            p_prev, p_cur = p_cur, p_next

        # Leapfrog carries two weakly-coupled sub-streams (the classic
        # odd/even decoupling). Blending the final pair at the readout is the
        # standard fix and costs nothing in reversibility, since both states
        # are held anyway. blend=0.5 averages them.
        final = blend * states[-1] + (1.0 - blend) * states[-2]
        return final, states

    def forward_hidden(self, idx):
        B, T = idx.shape
        assert T <= self.cfg.context, f"sequence {T} exceeds context {self.cfg.context}"
        p0 = self.embed(idx)
        cos, sin = self.rope(T, p0.device, p0.dtype)
        return self.run_stack(p0, cos, sin)

    def forward(self, idx, targets=None):
        final, _ = self.forward_hidden(idx)
        logits = self.lm_head(self.ln_f(final))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.reshape(-1)
        )
        return logits, loss


# ------------------------------------------------------------ data helper
def get_batch(data, batch_size, block_size, device, generator):
    """Sample a batch of (x, y) from a uint16 memmap. Shared by the scripts."""
    hi = len(data) - block_size - 1
    ix = torch.randint(hi, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def main():
    torch.manual_seed(SEED)
    print("=" * 72)
    print("MODEL — parameter counts")
    print("=" * 72)
    counts = {}
    for v in VARIANTS:
        torch.manual_seed(SEED)
        counts[v] = GPT(GPTConfig(variant=v)).num_params()

    n = counts["baseline"]
    emb = VOCAB_SIZE * D_MODEL
    per_block = (n - emb - 2 * D_MODEL) // N_LAYERS

    print()
    print("=" * 72)
    print("SUMMARY — record these")
    print("=" * 72)
    print(f"vocab / d_model / layers / heads : {VOCAB_SIZE} / {D_MODEL} / {N_LAYERS} / {N_HEADS}")
    print(f"head dim / d_ff / context        : {D_MODEL // N_HEADS} / {D_FF} / {CONTEXT}")
    print(f"dropout                          : {DROPOUT} (no Dropout modules exist)")
    print(f"positions                        : RoPE (0 params)")
    print(f"tied embeddings                  : True")
    print(f"h / blend                        : {H} / {BLEND}")
    print(f"embedding+head (tied)            : {emb:,}")
    print(f"per transformer block            : {per_block:,}")
    print(f"total params                     : {n:,} ({n / 1e6:.3f}M)")
    print(f"identical across variants        : {len(set(counts.values())) == 1}")
    print(f"within [{PARAM_MIN:,}, {PARAM_MAX:,}]  : {PARAM_MIN <= n <= PARAM_MAX}")
    print("=" * 72)


if __name__ == "__main__":
    main()
