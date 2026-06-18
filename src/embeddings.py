"""
WORD EMBEDDINGS
===============
Train a Word2Vec model on the TRAIN reviews only, then build the embedding
matrix that seeds the Keras Embedding layer.

The embedding matrix is indexed by Keras' word_index ids (offset by INDEX_FROM),
so row i corresponds to the same integer the padded sequences use. Words the
Word2Vec model never saw stay as zero vectors.

t-SNE visualisation of a handful of sentiment words is provided so the README
can *show* that the embedding space separates positive from negative language —
one of the tutorial's discussion questions, answered visually.
"""
from __future__ import annotations

import numpy as np
from gensim.models import Word2Vec

from .data import INDEX_FROM

# Sentiment-bearing probe words for the t-SNE sanity check.
PROBE_WORDS = [
    "great", "excellent", "wonderful", "brilliant", "love",
    "bad", "awful", "terrible", "boring", "worst",
    "film", "actor", "plot", "cast",
]


def train_word2vec(cfg: dict, tokens_train: list[list[str]]) -> Word2Vec:
    e = cfg["embeddings"]
    return Word2Vec(
        sentences=tokens_train,
        vector_size=e["vector_size"],
        window=e["window"],
        min_count=e["min_count"],
        sg=e["sg"],
        epochs=e["epochs"],
        workers=e["workers"],
        seed=cfg["project"]["random_state"],
    )


def build_embedding_matrix(cfg: dict, w2v: Word2Vec, word_index: dict) -> np.ndarray:
    """Map Keras word ids -> Word2Vec vectors. Returns (num_words, dim) matrix."""
    num_words = cfg["data"]["num_words"]
    dim = cfg["embeddings"]["vector_size"]
    matrix = np.zeros((num_words, dim), dtype="float32")
    hits = 0
    for word, idx in word_index.items():
        keras_id = idx + INDEX_FROM       # align with how data.py decodes ids
        if keras_id >= num_words:
            continue
        if word in w2v.wv:
            matrix[keras_id] = w2v.wv[word]
            hits += 1
    coverage = hits / min(num_words, len(word_index))
    print(f"[embed] matrix {matrix.shape}, vocab coverage = {coverage:.1%}")
    return matrix


def tsne_coordinates(w2v: Word2Vec, words: list[str] | None = None):
    """2-D t-SNE coords for probe words (returns words actually present + coords)."""
    from sklearn.manifold import TSNE

    words = words or PROBE_WORDS
    present = [w for w in words if w in w2v.wv]
    vecs = np.array([w2v.wv[w] for w in present])
    perplexity = max(2, min(len(present) - 1, 5))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(vecs)
    return present, coords
