"""
Autoresearch pretraining script — JAX/Flax/Optax port.
Single-device, single-file. Supports CUDA, CPU, and Apple Silicon (Metal).
Usage: uv run train.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict
from functools import partial
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax import struct

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device():
    try:
        devs = jax.devices("gpu")
        if devs:
            return "gpu"
    except Exception:
        pass
    try:
        devs = jax.devices("metal")
        if devs:
            return "metal"
    except Exception:
        pass
    return "cpu"

DEVICE_KIND = _select_device()

# Peak FLOPS for MFU calculation (0 = skip MFU reporting)
_DEVICE_PEAK_FLOPS = {
    "H100": 989.5e12,
    "A100": 312e12,
}

def _detect_peak_flops():
    if DEVICE_KIND not in ("gpu",):
        return 0
    try:
        name = jax.devices("gpu")[0].device_kind
        for key, val in _DEVICE_PEAK_FLOPS.items():
            if key in name:
                return val
    except Exception:
        pass
    return 312e12  # default to A100 if unknown GPU

PEAK_FLOPS = _detect_peak_flops()

# ---------------------------------------------------------------------------
# GPT Model (Flax)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def rms_norm(x):
    """RMS normalization along last dimension."""
    return x * jax.lax.rsqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6).astype(x.dtype)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    # x: (B, T, n_head, head_dim)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


def precompute_rotary_embeddings(seq_len, head_dim, base=10000):
    channel_range = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    # shape: (1, T, 1, head_dim//2) for broadcasting with (B, T, n_head, head_dim//2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return jnp.array(cos, dtype=jnp.bfloat16), jnp.array(sin, dtype=jnp.bfloat16)


def compute_window_sizes(config):
    pattern = config.window_pattern.upper()
    assert all(c in "SL" for c in pattern)
    long_window = config.sequence_len
    short_window = long_window // 2
    char_to_window = {"L": long_window, "S": short_window}
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    window_sizes[-1] = long_window
    return window_sizes


def make_causal_mask(T, window_size):
    """Boolean mask (T, T): True where attention is allowed."""
    i = jnp.arange(T)[:, None]
    j = jnp.arange(T)[None, :]
    causal = i >= j
    windowed = (i - j) < window_size
    return causal & windowed


class CausalSelfAttention(nn.Module):
    config: GPTConfig
    layer_idx: int

    @nn.compact
    def __call__(self, x, ve, cos, sin, window_size):
        cfg = self.config
        B, T, C = x.shape
        head_dim = cfg.n_embd // cfg.n_head

        q = nn.Dense(cfg.n_head * head_dim, use_bias=False, name="c_q")(x)
        k = nn.Dense(cfg.n_kv_head * head_dim, use_bias=False, name="c_k")(x)
        v = nn.Dense(cfg.n_kv_head * head_dim, use_bias=False, name="c_v")(x)

        q = q.reshape(B, T, cfg.n_head, head_dim)
        k = k.reshape(B, T, cfg.n_kv_head, head_dim)
        v = v.reshape(B, T, cfg.n_kv_head, head_dim)

        # Value residual (ResFormer gate)
        if has_ve(self.layer_idx, cfg.n_layer):
            ve_gate_channels = 32
            ve = ve.reshape(B, T, cfg.n_kv_head, head_dim)
            gate = 2 * jax.nn.sigmoid(
                nn.Dense(cfg.n_kv_head, use_bias=False, name="ve_gate")(x[..., :ve_gate_channels])
            )  # (B, T, n_kv_head)
            v = v + gate[..., None] * ve

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        # GQA: repeat k/v heads to match q heads
        if cfg.n_kv_head < cfg.n_head:
            repeat = cfg.n_head // cfg.n_kv_head
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Windowed causal attention via mask
        # q/k/v: (B, T, n_head, head_dim) -> need (B, n_head, T, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        mask = make_causal_mask(T, window_size)  # (T, T)
        # scale
        scale = head_dim ** -0.5
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        attn = jnp.where(mask[None, None, :, :], attn, jnp.finfo(q.dtype).min)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
        y = jnp.einsum("bhij,bhjd->bhid", attn, v)  # (B, n_head, T, head_dim)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)

        y = nn.Dense(cfg.n_embd, use_bias=False, name="c_proj")(y)
        return y


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.config.n_embd, use_bias=False, name="c_fc")(x)
        x = jax.nn.relu(x) ** 2
        x = nn.Dense(self.config.n_embd, use_bias=False, name="c_proj")(x)
        return x


class Block(nn.Module):
    config: GPTConfig
    layer_idx: int

    @nn.compact
    def __call__(self, x, ve, cos, sin, window_size):
        x = x + CausalSelfAttention(self.config, self.layer_idx, name="attn")(
            rms_norm(x), ve, cos, sin, window_size
        )
        x = x + MLP(self.config, name="mlp")(rms_norm(x))
        return x


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        cfg = self.config
        self.wte = nn.Embed(cfg.vocab_size, cfg.n_embd, name="wte")
        self.blocks = [Block(cfg, i, name=f"h_{i}") for i in range(cfg.n_layer)]
        self.lm_head = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head")
        # Per-layer scalars
        self.resid_lambdas = self.param("resid_lambdas", nn.initializers.ones, (cfg.n_layer,))
        self.x0_lambdas = self.param("x0_lambdas", lambda rng, shape: jnp.full(shape, 0.1), (cfg.n_layer,))
        # Value embeddings for alternating layers
        self.value_embeds = {
            str(i): nn.Embed(cfg.vocab_size, cfg.n_kv_head * (cfg.n_embd // cfg.n_head), name=f"ve_{i}")
            for i in range(cfg.n_layer) if has_ve(i, cfg.n_layer)
        }

    def __call__(self, idx, cos, sin, window_sizes):
        B, T = idx.shape
        x = self.wte(idx).astype(jnp.bfloat16)
        x = rms_norm(x)
        x0 = x

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).astype(jnp.bfloat16) if str(i) in self.value_embeds else None
            x = block(x, ve, cos[:, :T], sin[:, :T], window_sizes[i])

        x = rms_norm(x)
        softcap = 15.0
        logits = self.lm_head(x).astype(jnp.float32)
        logits = softcap * jnp.tanh(logits / softcap)
        return logits


def init_params(model, config, rng, rotary_seq_len):
    """Initialize model parameters with custom weight init."""
    head_dim = config.n_embd // config.n_head
    cos, sin = precompute_rotary_embeddings(rotary_seq_len, head_dim)
    window_sizes = compute_window_sizes(config)

    dummy_idx = jnp.zeros((1, config.sequence_len), dtype=jnp.int32)
    params = model.init(rng, dummy_idx, cos, sin, window_sizes)

    # Custom weight initialization
    import jax.tree_util as jtu

    def reinit(params):
        n_embd = config.n_embd
        s = (3 ** 0.5) * (n_embd ** -0.5)
        rng_seq = jax.random.split(jax.random.PRNGKey(42), 1000)
        idx = [0]

        def next_rng():
            r = rng_seq[idx[0]]
            idx[0] += 1
            return r

        p = params["params"]

        # Embedding
        p["wte"]["embedding"] = jax.random.normal(next_rng(), p["wte"]["embedding"].shape).astype(jnp.bfloat16)

        # lm_head
        p["lm_head"]["kernel"] = (jax.random.normal(next_rng(), p["lm_head"]["kernel"].shape) * 0.001)

        # Per-layer scalars
        p["resid_lambdas"] = jnp.ones_like(p["resid_lambdas"])
        p["x0_lambdas"] = jnp.full_like(p["x0_lambdas"], 0.1)

        # Transformer blocks
        for i in range(config.n_layer):
            blk = p[f"h_{i}"]
            attn = blk["attn"]
            mlp = blk["mlp"]
            for name in ["c_q", "c_k", "c_v"]:
                k = attn[name]["kernel"]
                attn[name]["kernel"] = jax.random.uniform(next_rng(), k.shape, minval=-s, maxval=s)
            attn["c_proj"]["kernel"] = jnp.zeros_like(attn["c_proj"]["kernel"])
            if "ve_gate" in attn:
                attn["ve_gate"]["kernel"] = jnp.zeros_like(attn["ve_gate"]["kernel"])
            mlp["c_fc"]["kernel"] = jax.random.uniform(next_rng(), mlp["c_fc"]["kernel"].shape, minval=-s, maxval=s)
            mlp["c_proj"]["kernel"] = jnp.zeros_like(mlp["c_proj"]["kernel"])

        # Value embeddings
        for i in range(config.n_layer):
            ve_key = f"ve_{i}"
            if ve_key in p:
                emb = p[ve_key]["embedding"]
                p[ve_key]["embedding"] = jax.random.uniform(
                    next_rng(), emb.shape, minval=-s, maxval=s
                ).astype(jnp.bfloat16)

        return params

    params = reinit(params)
    return params, cos, sin, window_sizes


# ---------------------------------------------------------------------------
# Parameter counting / FLOPs
# ---------------------------------------------------------------------------

def count_params(params, config):
    """Return dict of parameter counts by group."""
    import jax.tree_util as jtu
    p = params["params"]
    wte = p["wte"]["embedding"].size
    lm_head = p["lm_head"]["kernel"].size
    scalars = p["resid_lambdas"].size + p["x0_lambdas"].size
    value_embeds = sum(
        p[f"ve_{i}"]["embedding"].size
        for i in range(config.n_layer) if f"ve_{i}" in p
    )
    transformer_matrices = sum(
        leaf.size for key, leaf in jtu.tree_leaves_with_path(p)
        if any(f"h_{i}" in str(key) for i in range(config.n_layer))
    )
    total = wte + lm_head + scalars + value_embeds + transformer_matrices
    return {
        "wte": wte, "value_embeds": value_embeds, "lm_head": lm_head,
        "transformer_matrices": transformer_matrices, "scalars": scalars, "total": total,
    }


def estimate_flops(config, window_sizes):
    """Estimated FLOPs per token (forward + backward, 6x rule + attention)."""
    # Count non-embedding params
    n_embd = config.n_embd
    n_layer = config.n_layer
    n_head = config.n_head
    n_kv_head = config.n_kv_head
    head_dim = n_embd // n_head

    # Attention matrices per layer
    attn_params = (n_head + 2 * n_kv_head) * head_dim * n_embd + n_embd * n_embd
    mlp_params = 2 * 4 * n_embd * n_embd
    # VE gate + VE embed (approximate)
    ve_per_layer = n_kv_head * head_dim * 32 if True else 0

    matrix_params = n_layer * (attn_params + mlp_params)
    attn_flops = 0
    for w in window_sizes:
        effective_seq = min(w, config.sequence_len)
        attn_flops += 12 * n_head * head_dim * effective_seq

    return 6 * matrix_params + attn_flops


# ---------------------------------------------------------------------------
# Optimizer: MuonAdamW via Optax
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(X, ns_steps):
    """Newton-Schulz orthogonalization in JAX."""
    X = X / (jnp.linalg.norm(X, ord='fro') * 1.02 + 1e-6)
    X = X.astype(jnp.bfloat16)
    rows, cols = X.shape[-2], X.shape[-1]
    if rows >= cols:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.T @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def muon_update(grad, momentum_buf, second_mom_buf, momentum, beta2, ns_steps, lr_scale):
    """Single Muon step. Returns updated grad, new momentum_buf, new second_mom_buf."""
    # Nesterov momentum
    new_momentum_buf = momentum * momentum_buf + (1 - momentum) * grad
    g = grad + momentum * (new_momentum_buf - momentum_buf)

    # Polar express orthogonalization
    g_orth = polar_express(g, ns_steps)

    # NorMuon variance reduction
    rows, cols = g_orth.shape[-2], g_orth.shape[-1]
    red_dim = -1 if rows >= cols else -2
    v_mean = g_orth.astype(jnp.float32) ** 2
    if red_dim == -1:
        v_mean = v_mean.mean(axis=-1, keepdims=True)
        red_dim_size = cols
    else:
        v_mean = v_mean.mean(axis=-2, keepdims=True)
        red_dim_size = rows

    new_second_mom = beta2 * second_mom_buf + (1 - beta2) * v_mean.astype(second_mom_buf.dtype)
    step_size = jnp.maximum(new_second_mom, 1e-10) ** -0.5

    v_norm_sq = (v_mean * red_dim_size).sum()
    v_norm = jnp.sqrt(v_norm_sq)
    scaled_sq_sum = (v_mean * red_dim_size * step_size.astype(jnp.float32) ** 2).sum()
    v_norm_new = jnp.sqrt(jnp.maximum(scaled_sq_sum, 1e-20))
    final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))

    g_out = (g_orth * final_scale.astype(g_orth.dtype)) * lr_scale
    return g_out, new_momentum_buf, new_second_mom


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

class TrainState:
    """Simple training state container."""
    def __init__(self, params, opt_state, step):
        self.params = params
        self.opt_state = opt_state
        self.step = step


def build_param_labels(params, config):
    """
    Label each parameter leaf as one of:
      'adamw_emb', 'adamw_lm', 'adamw_ve', 'adamw_scalar', 'muon_<shape>'
    Returns a pytree of the same structure as params with string labels.
    """
    import jax.tree_util as jtu

    def label_fn(path, leaf):
        path_str = "/".join(str(k) for k in path)
        if "wte" in path_str:
            return "adamw_emb"
        if "lm_head" in path_str:
            return "adamw_lm"
        if path_str.startswith("params/ve_") or "/ve_" in path_str:
            return "adamw_ve"
        if "resid_lambdas" in path_str or "x0_lambdas" in path_str:
            return "adamw_scalar"
        # All block parameters (2D matrices) -> Muon
        if leaf.ndim == 2:
            return f"muon_{leaf.shape[0]}x{leaf.shape[1]}"
        # 1D or other (bias-free, but just in case)
        return "adamw_scalar"

    return jtu.tree_map_with_path(label_fn, params)


def setup_optimizer(params, config,
                    unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                    weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
    """Build optax optimizer with per-parameter-group learning rates."""
    model_dim = config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

    b1, b2 = adam_betas

    def make_adamw(lr, wd=0.0, betas=(b1, b2)):
        return optax.adamw(lr, b1=betas[0], b2=betas[1], eps=1e-10, weight_decay=wd)

    # Collect unique matrix shapes for Muon groups
    import jax.tree_util as jtu
    matrix_shapes = set()
    for path, leaf in jtu.tree_leaves_with_path(params):
        path_str = "/".join(str(k) for k in path)
        if leaf.ndim == 2:
            is_emb = "wte" in path_str or "ve_" in path_str
            is_lm = "lm_head" in path_str
            if not is_emb and not is_lm:
                matrix_shapes.add(leaf.shape)

    # Build label -> tx mapping
    tx_map = {
        "adamw_lm":     make_adamw(unembedding_lr * dmodel_lr_scale),
        "adamw_emb":    make_adamw(embedding_lr * dmodel_lr_scale),
        "adamw_ve":     make_adamw(embedding_lr * dmodel_lr_scale),
        "adamw_scalar_resid": make_adamw(scalar_lr * 0.01),
        "adamw_scalar_x0":    make_adamw(scalar_lr, betas=(0.96, 0.95)),
        "adamw_scalar": make_adamw(scalar_lr * 0.01),
    }
    for shape in matrix_shapes:
        lr_scaled = matrix_lr * max(1.0, shape[0] / shape[1]) ** 0.5
        tx_map[f"muon_{shape[0]}x{shape[1]}"] = optax.sgd(lr_scaled, momentum=0.0)  # placeholder

    def label_fn_full(path, leaf):
        path_str = "/".join(str(k) for k in path)
        if "wte" in path_str:
            return "adamw_emb"
        if "lm_head" in path_str:
            return "adamw_lm"
        if "/ve_" in path_str or path_str.startswith("params/ve_"):
            return "adamw_ve"
        if "resid_lambdas" in path_str:
            return "adamw_scalar_resid"
        if "x0_lambdas" in path_str:
            return "adamw_scalar_x0"
        if leaf.ndim == 2:
            return f"muon_{leaf.shape[0]}x{leaf.shape[1]}"
        return "adamw_scalar"

    param_labels = jtu.tree_map_with_path(label_fn_full, params)
    tx = optax.multi_transform(tx_map, param_labels)
    opt_state = tx.init(params)
    return tx, opt_state, matrix_shapes


# ---------------------------------------------------------------------------
# Muon state (kept separately from optax state for simplicity)
# ---------------------------------------------------------------------------

def init_muon_state(params, matrix_shapes):
    """Initialize Muon momentum buffers."""
    import jax.tree_util as jtu
    muon_state = {}
    for shape in matrix_shapes:
        muon_state[shape] = {
            "momentum": jnp.zeros(shape),
            "second_mom": jnp.zeros((shape[0], 1) if shape[0] >= shape[1] else (1, shape[1])),
        }
    return muon_state


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

DEPTH = 8
DEVICE_BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# LR / schedule helpers
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
key = jax.random.PRNGKey(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")
print(f"JAX backend: {jax.default_backend()} ({DEVICE_KIND})")


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
rotary_seq_len = config.sequence_len * 10
params, cos, sin, window_sizes = init_params(model, config, key, rotary_seq_len)

param_counts = count_params(params, config)
print("Parameter counts:")
for key_name, value in param_counts.items():
    print(f"  {key_name:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = estimate_flops(config, window_sizes)
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# Optimizer
tx, opt_state, matrix_shapes = setup_optimizer(
    params, config,
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)
muon_state = init_muon_state(params, matrix_shapes)

state = TrainState(params=params, opt_state=opt_state, step=0)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# JIT-compiled forward + grad
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=())
def forward_loss(params, x, y, cos, sin, window_sizes_tuple):
    logits = model.apply(params, x, cos, sin, list(window_sizes_tuple))
    # cross entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    y_flat = y.reshape(-1)
    lp_flat = log_probs.reshape(-1, logits.shape[-1])
    loss = -lp_flat[jnp.arange(len(y_flat)), y_flat].mean()
    return loss


grad_fn = jax.jit(jax.value_and_grad(
    lambda params, x, y: forward_loss(params, x, y, cos, sin, tuple(window_sizes))
))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0
window_sizes_tuple = tuple(window_sizes)

while True:
    jax.effects_barrier()
    t0 = time.time()

    # Gradient accumulation
    accumulated_grads = None
    train_loss = None
    for micro_step in range(grad_accum_steps):
        loss, grads = grad_fn(state.params, x, y)
        if accumulated_grads is None:
            accumulated_grads = grads
            train_loss = loss
        else:
            accumulated_grads = jax.tree_util.tree_map(
                lambda a, b: a + b, accumulated_grads, grads
            )
        x, y, epoch = next(train_loader)

    # Scale grads by 1/grad_accum_steps
    accumulated_grads = jax.tree_util.tree_map(
        lambda g: g / grad_accum_steps, accumulated_grads
    )

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_wd = get_weight_decay(progress)

    # Scale optax optimizer learning rates via a multiplier
    # We rebuild the tx with scaled LRs by injecting a scale transform
    scaled_updates, new_opt_state = tx.update(accumulated_grads, state.opt_state, state.params)

    # Apply Muon updates manually for matrix params
    import jax.tree_util as jtu

    def apply_updates_with_muon(params, grads, opt_updates):
        """Replace optax updates for matrix params with Muon updates."""
        def update_leaf(path, param, grad, opt_upd):
            path_str = "/".join(str(k) for k in path)
            is_emb = "wte" in path_str or "/ve_" in path_str or path_str.startswith("params/ve_")
            is_lm = "lm_head" in path_str
            is_scalar = "resid_lambdas" in path_str or "x0_lambdas" in path_str
            if param.ndim == 2 and not is_emb and not is_lm and not is_scalar:
                shape = param.shape
                ms = muon_state[shape]
                lr_scale = MATRIX_LR * max(1.0, shape[0] / shape[1]) ** 0.5 * lrm
                g_out, new_mom, new_second = muon_update(
                    grad, ms["momentum"], ms["second_mom"],
                    muon_momentum, 0.95, 5, lr_scale
                )
                muon_state[shape]["momentum"] = new_mom
                muon_state[shape]["second_mom"] = new_second
                # Cautious weight decay
                mask = (g_out * param) >= 0
                update = -g_out - lr_scale * muon_wd * param * mask
                return param + update
            else:
                # Use optax update, scaled by lrm
                return param + opt_upd * lrm
        return jtu.tree_map_with_path(update_leaf, params, grads, opt_updates)

    new_params = apply_updates_with_muon(state.params, accumulated_grads, scaled_updates)
    state = TrainState(params=new_params, opt_state=new_opt_state, step=state.step + 1)

    train_loss_f = float(train_loss)

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    jax.effects_barrier()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    mfu_str = ""
    if PEAK_FLOPS > 0:
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_FLOPS
        mfu_str = f" | mfu: {mfu:.1f}%"

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | "
        f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,}{mfu_str} | epoch: {epoch} | remaining: {remaining:.0f}s    ",
        end="", flush=True
    )

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
def model_apply(params, x):
    return model.apply(params, x, cos, sin, window_sizes)

val_bpb = evaluate_bpb(model_apply, state.params, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / PEAK_FLOPS
    if total_training_time > 0 and PEAK_FLOPS > 0 else 0
)

# Memory stats
peak_mem_mb = 0.0
try:
    mem_stats = jax.devices()[0].memory_stats()
    peak_mem_mb = mem_stats.get("peak_bytes_in_use", 0) / 1024 / 1024
except Exception:
    pass

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_mem_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
