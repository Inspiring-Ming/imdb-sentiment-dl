"""
DATA LAYER
==========
Load and decode the Keras IMDb dataset, then carve a *validation* split out of
the training set.

Why this differs from the tutorial:

  The tutorial trains Word2Vec on the decoded TRAIN reviews and then evaluates
  the classifier on the TEST reviews — but it never holds out a validation set
  from train, so every hyper-parameter choice is implicitly tuned on the test
  set. Here we keep three disjoint splits:

      train  -> fit Word2Vec + fit the network
      val    -> model selection / early stopping signal
      test   -> touched exactly once, for the final reported number

This is the single most important habit that separates a shipped model from a
notebook number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from keras.datasets import imdb
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

# Keras reserves 0,1,2 for <pad>, <start>, <oov>; real words start at index 3.
INDEX_FROM = 3


@dataclass
class Dataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    tokens_train: list[list[str]]   # decoded train reviews (for Word2Vec)
    word_index: dict                # word -> integer id (offset by INDEX_FROM)


def _decode(seqs, reverse_index) -> list[list[str]]:
    """Turn integer sequences back into token lists for embedding training."""
    out = []
    for seq in seqs:
        out.append([reverse_index.get(i - INDEX_FROM, "<oov>") for i in seq])
    return out


def load_dataset(cfg: dict) -> Dataset:
    num_words = cfg["data"]["num_words"]
    maxlen = cfg["data"]["maxlen"]
    rs = cfg["project"]["random_state"]

    (X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(num_words=num_words)

    # Hold out a validation set BEFORE doing anything else (no leakage).
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train,
        test_size=cfg["data"]["val_size"],
        random_state=rs, stratify=y_train,
    )

    word_index = imdb.get_word_index()
    reverse_index = {v: k for k, v in word_index.items()}
    tokens_train = _decode(X_tr_raw, reverse_index)   # Word2Vec sees TRAIN only

    # Pad to a fixed length so the network gets rectangular input.
    X_train = pad_sequences(X_tr_raw, maxlen=maxlen)
    X_val = pad_sequences(X_val_raw, maxlen=maxlen)
    X_test = pad_sequences(X_test_raw, maxlen=maxlen)

    return Dataset(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=np.asarray(y_tr), y_val=np.asarray(y_val), y_test=np.asarray(y_test),
        tokens_train=tokens_train, word_index=word_index,
    )
