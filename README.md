# autoresearch (JAX port)

> **This is a JAX/Flax/Optax port of [autoresearch](https://github.com/karpathy/autoresearch) by [@karpathy](https://github.com/karpathy).**
> The original codebase, experiment design, training loop, evaluation metric, and agent workflow are his work.
> This port adapts the implementation to run on JAX — removing the CUDA-only Flash Attention 3 dependency
> and enabling Apple Silicon (Metal) and CPU support.

Runs on NVIDIA CUDA, Apple Silicon (Metal), and CPU — no Flash Attention kernel required.

The experiment loop, metric (`val_bpb`), and agent workflow are identical to the original.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install JAX for your platform (choose one):
#    NVIDIA CUDA 12:
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#    Apple Silicon:
pip install jax-metal
#    CPU only:
pip install "jax[cpu]"

# 3. Install project dependencies
uv sync

# 4. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 5. Run a single training experiment (~5 min)
uv run train.py
```

For CPU / Apple Silicon, reduce batch size for faster iteration:
edit `TOTAL_BATCH_SIZE = 2**14` and `DEVICE_BATCH_SIZE = 16` in `train.py`.

## Platform support

| Platform | JAX package | Notes |
|---|---|---|
| NVIDIA (CUDA 12) | `jax[cuda12]` | Full speed, MFU reported |
| Apple Silicon | `jax-metal` | Metal GPU acceleration |
| CPU | `jax[cpu]` | Slow but works |

## How it works

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model (Flax), optimizer (Muon + AdamW via Optax), and training loop. Everything is fair game.
- **`program.md`** — baseline instructions for one agent.

Training runs for a **fixed 5-minute time budget**. The metric is **val_bpb** (lower is better).

## Key differences from PyTorch original

| Concern | PyTorch original | This JAX port |
|---|---|---|
| NN framework | `torch.nn.Module` | `flax.linen.Module` |
| Attention | Flash Attention 3 (CUDA-only) | `jnp.einsum` + causal mask |
| Windowed attn | FA3 `window_size` param | Boolean causal+local mask |
| Optimizer | Custom `MuonAdamW` class | `optax.multi_transform` + Muon |
| JIT | `torch.compile` | `jax.jit` |
| bfloat16 | `torch.amp.autocast` | explicit `.astype(jnp.bfloat16)` |
| Token bytes | `torch.save/load` | `np.save/load` |
| Memory stats | `torch.cuda.max_memory_allocated` | `jax.devices()[0].memory_stats()` |
| Random seed | `torch.manual_seed(42)` | `jax.random.PRNGKey(42)` |

## Running tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v -k "not integration"  # skip data-dependent tests
```

## License

MIT
