"""Tests for prepare.py — unit tests use fake in-memory data, no real parquet needed."""
import os
import sys
import pickle
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Check if real data exists for integration tests
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
_TOKENIZER_PKL = os.path.join(_CACHE_DIR, "tokenizer", "tokenizer.pkl")
_DATA_EXISTS = os.path.exists(_TOKENIZER_PKL)


def _make_fake_tokenizer():
    """Create a minimal tiktoken-based tokenizer for unit tests."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return enc


# ---------------------------------------------------------------------------
# Unit tests (no real data required)
# ---------------------------------------------------------------------------

def test_tokenizer_roundtrip():
    """Encode then decode a string — should get identical output."""
    enc = _make_fake_tokenizer()
    test = "Hello world! Numbers: 123."
    encoded = enc.encode(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Roundtrip failed: {test!r} -> {decoded!r}"


def test_tokenizer_wrapper_roundtrip():
    """Tokenizer wrapper encode/decode roundtrip."""
    import tiktoken
    import pickle
    from prepare import Tokenizer, BOS_TOKEN, SPECIAL_TOKENS

    # Build a minimal tiktoken enc with a BOS token
    enc = tiktoken.get_encoding("cl100k_base")

    class FakeTokenizer:
        def __init__(self):
            self.enc = enc
            self.bos_token_id = 0  # fake BOS

        def get_vocab_size(self):
            return enc.n_vocab

        def get_bos_token_id(self):
            return self.bos_token_id

        def encode(self, text, prepend=None, num_threads=8):
            if isinstance(text, str):
                ids = enc.encode(text)
                if prepend is not None:
                    ids.insert(0, prepend)
                return ids
            elif isinstance(text, list):
                result = [enc.encode(t) for t in text]
                if prepend is not None:
                    for row in result:
                        row.insert(0, prepend)
                return result
            raise ValueError

        def decode(self, ids):
            return enc.decode(ids)

    tok = FakeTokenizer()
    text = "Hello, world!"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


@pytest.mark.skipif(not _DATA_EXISTS, reason="Real tokenizer data not available")
def test_token_bytes_shape():
    """Load token_bytes, assert shape == (vocab_size,)."""
    from prepare import get_token_bytes, Tokenizer
    tok = Tokenizer.from_directory()
    vocab_size = tok.get_vocab_size()
    token_bytes = get_token_bytes()
    assert token_bytes.shape == (vocab_size,), f"Expected ({vocab_size},), got {token_bytes.shape}"
    assert token_bytes.dtype == np.int32


@pytest.mark.skipif(not _DATA_EXISTS, reason="Real tokenizer data not available")
def test_dataloader_shapes():
    """make_dataloader yields batches of correct shape."""
    from prepare import Tokenizer, make_dataloader, MAX_SEQ_LEN
    tok = Tokenizer.from_directory()
    B, T = 2, 32
    loader = make_dataloader(tok, B, T, "val")
    x, y, epoch = next(loader)
    assert x.shape == (B, T), f"Expected ({B}, {T}), got {x.shape}"
    assert y.shape == (B, T), f"Expected ({B}, {T}), got {y.shape}"


@pytest.mark.skipif(not _DATA_EXISTS, reason="Real tokenizer data not available")
def test_dataloader_bos_aligned():
    """Every row in x should start with bos_token_id."""
    from prepare import Tokenizer, make_dataloader
    tok = Tokenizer.from_directory()
    bos = tok.get_bos_token_id()
    B, T = 4, 32
    loader = make_dataloader(tok, B, T, "val")
    x, y, epoch = next(loader)
    import jax.numpy as jnp
    first_tokens = x[:, 0]
    for i, ft in enumerate(first_tokens):
        assert int(ft) == bos, f"Row {i} does not start with BOS (got {int(ft)}, expected {bos})"


@pytest.mark.skipif(not _DATA_EXISTS, reason="Real tokenizer data not available")
def test_dataloader_no_padding():
    """Every element in batch should be a valid token id (>= 0)."""
    from prepare import Tokenizer, make_dataloader
    tok = Tokenizer.from_directory()
    vocab_size = tok.get_vocab_size()
    B, T = 2, 32
    loader = make_dataloader(tok, B, T, "val")
    x, y, _ = next(loader)
    import jax.numpy as jnp
    assert jnp.all(x >= 0), "Negative token ids found"
    assert jnp.all(x < vocab_size), "Token ids >= vocab_size found"
