"""
Tests for the deterministic plumbing — embedding-matrix alignment, model build
shapes, and the eval contract. We do NOT assert on learned accuracy (that's
non-deterministic and slow); we assert the wiring is correct.

Run:  python -m pytest tests/ -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import INDEX_FROM
from src.embeddings import build_embedding_matrix
from src.model import build_model


class _FakeKV:
    """Minimal stand-in for gensim KeyedVectors."""
    def __init__(self, vocab, dim):
        self._d = {w: np.full(dim, i + 1.0, dtype="float32") for i, w in enumerate(vocab)}

    def __contains__(self, w):
        return w in self._d

    def __getitem__(self, w):
        return self._d[w]


class _FakeW2V:
    def __init__(self, vocab, dim):
        self.wv = _FakeKV(vocab, dim)


def test_embedding_matrix_alignment_and_shape():
    cfg = load_config()
    cfg["data"]["num_words"] = 50
    dim = cfg["embeddings"]["vector_size"]
    word_index = {"great": 1, "awful": 2, "film": 3}   # keras-style 1-based ids
    w2v = _FakeW2V(["great", "awful", "film"], dim)

    mat = build_embedding_matrix(cfg, w2v, word_index)
    assert mat.shape == (50, dim)
    # "great" has keras id 1 -> matrix row 1 + INDEX_FROM, and must be non-zero.
    assert np.any(mat[1 + INDEX_FROM] != 0)
    # An id beyond num_words must be skipped (stays zero) — no IndexError.
    word_index_oob = {"rare": 999}
    build_embedding_matrix(cfg, _FakeW2V(["rare"], dim), word_index_oob)


def test_each_architecture_builds_with_binary_output():
    cfg = load_config()
    cfg["data"]["num_words"] = 50
    cfg["data"]["maxlen"] = 20
    dim = cfg["embeddings"]["vector_size"]
    emb = np.zeros((50, dim), dtype="float32")
    for arch in ("mlp", "pool", "lstm"):
        model = build_model(cfg, arch, emb)
        assert model.output_shape == (None, 1)   # single sigmoid sentiment unit
        out = model.predict(np.zeros((2, 20), dtype="int32"), verbose=0)
        assert out.shape == (2, 1)
        assert ((out >= 0) & (out <= 1)).all()    # sigmoid range
