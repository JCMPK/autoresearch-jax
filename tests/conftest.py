"""Shared fixtures for autoresearch-jax tests."""
import sys
import os

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from train import GPTConfig, GPT, compute_window_sizes, precompute_rotary_embeddings, init_params


@pytest.fixture(scope="session")
def tiny_config():
    return GPTConfig(
        n_layer=2,
        n_embd=64,
        n_head=2,
        n_kv_head=2,
        sequence_len=32,
        vocab_size=256,
        window_pattern="SL",
    )


@pytest.fixture(scope="session")
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture(scope="session")
def tiny_batch(tiny_config):
    key = jax.random.PRNGKey(1)
    return jax.random.randint(key, shape=(2, tiny_config.sequence_len), minval=0, maxval=tiny_config.vocab_size)


@pytest.fixture(scope="session")
def tiny_model_and_params(tiny_config, rng):
    model = GPT(tiny_config)
    params, cos, sin, window_sizes = init_params(model, tiny_config, rng, tiny_config.sequence_len * 2)
    return model, params, cos, sin, window_sizes
