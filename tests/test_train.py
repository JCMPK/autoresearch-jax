"""Tests for train.py — all run on CPU, no GPU required."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax

from train import (
    GPTConfig, GPT, rms_norm, apply_rotary_emb, has_ve,
    precompute_rotary_embeddings, compute_window_sizes, make_causal_mask,
    init_params, polar_express, get_lr_multiplier,
    WARMUP_RATIO, WARMDOWN_RATIO,
)


def test_model_forward_shape(tiny_model_and_params, tiny_batch, tiny_config):
    model, params, cos, sin, window_sizes = tiny_model_and_params
    logits = model.apply(params, tiny_batch, cos, sin, window_sizes)
    B, T = tiny_batch.shape
    assert logits.shape == (B, T, tiny_config.vocab_size)


def test_model_loss_finite(tiny_model_and_params, tiny_batch, tiny_config):
    model, params, cos, sin, window_sizes = tiny_model_and_params
    logits = model.apply(params, tiny_batch, cos, sin, window_sizes)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    y = tiny_batch
    y_flat = y.reshape(-1)
    lp_flat = log_probs.reshape(-1, logits.shape[-1])
    loss = -lp_flat[jnp.arange(len(y_flat)), y_flat].mean()
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"


def test_model_no_nan_after_step(tiny_model_and_params, tiny_batch, tiny_config):
    model, params, cos, sin, window_sizes = tiny_model_and_params
    x = tiny_batch
    y = tiny_batch

    def loss_fn(p):
        logits = model.apply(p, x, cos, sin, window_sizes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        y_flat = y.reshape(-1)
        lp_flat = log_probs.reshape(-1, logits.shape[-1])
        return -lp_flat[jnp.arange(len(y_flat)), y_flat].mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)
    updates, _ = tx.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    # Check all params are finite
    leaves = jax.tree_util.tree_leaves(new_params)
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf)), f"Non-finite params after step: {leaf}"


def test_rotary_emb_shape(tiny_config):
    head_dim = tiny_config.n_embd // tiny_config.n_head
    T = tiny_config.sequence_len
    cos, sin = precompute_rotary_embeddings(T, head_dim)
    B, n_head = 2, tiny_config.n_head
    x = jnp.ones((B, T, n_head, head_dim))
    out = apply_rotary_emb(x, cos, sin)
    assert out.shape == x.shape


def test_rms_norm_unit_scale():
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (4, 16, 64))
    y = rms_norm(x)
    # RMS of y along last dim should be ~1
    rms = jnp.sqrt(jnp.mean(y.astype(jnp.float32) ** 2, axis=-1))
    np.testing.assert_allclose(rms, 1.0, atol=1e-4)


def test_muon_orthogonal():
    key = jax.random.PRNGKey(7)
    W = jax.random.normal(key, (32, 16))
    W_orth = polar_express(W, ns_steps=5)
    # W_orth should satisfy W^T W ≈ I (for tall matrices)
    WtW = W_orth.T @ W_orth
    I = jnp.eye(16)
    np.testing.assert_allclose(WtW, I, atol=0.1)


def test_lr_schedule():
    # warmup region
    if WARMUP_RATIO > 0:
        lrm = get_lr_multiplier(WARMUP_RATIO / 2)
        assert 0 < lrm < 1.0
    # plateau
    lrm = get_lr_multiplier(0.5 * (1.0 - WARMDOWN_RATIO))
    assert lrm == pytest.approx(1.0, abs=1e-6)
    # warmdown
    lrm = get_lr_multiplier(1.0)
    assert lrm >= 0.0


def test_window_mask_causal():
    T = 16
    window_size = 8
    mask = make_causal_mask(T, window_size)
    # Must be causal (upper triangle is False)
    for i in range(T):
        for j in range(T):
            if j > i:
                assert not mask[i, j], f"mask[{i},{j}] should be False (not causal)"
            if i - j >= window_size:
                assert not mask[i, j], f"mask[{i},{j}] should be False (outside window)"
            if 0 <= i - j < window_size:
                assert mask[i, j], f"mask[{i},{j}] should be True"


def test_full_train_step_cpu(tiny_config, rng, tiny_batch):
    """Run 3 train steps on tiny model, assert loss stays finite."""
    model = GPT(tiny_config)
    params, cos, sin, window_sizes = init_params(model, tiny_config, rng, tiny_config.sequence_len * 2)

    x = tiny_batch
    y = tiny_batch

    def loss_fn(p):
        logits = model.apply(p, x, cos, sin, window_sizes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        y_flat = y.reshape(-1)
        lp_flat = log_probs.reshape(-1, logits.shape[-1])
        return -lp_flat[jnp.arange(len(y_flat)), y_flat].mean()

    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    losses = []
    for _ in range(3):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        losses.append(float(loss))
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    for loss in losses:
        assert jnp.isfinite(loss), f"Loss not finite: {loss}"
