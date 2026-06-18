"""
MODEL ARCHITECTURES
===================
Three Keras architectures over the same Word2Vec embedding layer, so we can run
a bake-off instead of accepting the first thing that trains:

  "mlp"   Embedding -> Flatten -> Dense -> Dropout -> sigmoid   (the tutorial)
  "pool"  Embedding -> GlobalAveragePooling1D -> Dense -> ...    (order-invariant,
            far fewer params than flattening maxlen*dim features)
  "lstm"  Embedding -> LSTM -> Dense -> ...                      (sequence-aware)

The tutorial's `Flatten` produces maxlen*vector_size (e.g. 300*100 = 30,000)
inputs into the first Dense layer — millions of weights that mostly memorise
position. Global pooling and the LSTM are the standard fixes; the bake-off shows
which actually wins on validation.
"""
from __future__ import annotations

import numpy as np
from keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    Input,
)
from keras.models import Sequential


def build_model(cfg: dict, arch: str, embedding_matrix: np.ndarray) -> Sequential:
    num_words = cfg["data"]["num_words"]
    maxlen = cfg["data"]["maxlen"]
    dim = cfg["embeddings"]["vector_size"]
    m = cfg["model"]

    emb = Embedding(
        input_dim=num_words,
        output_dim=dim,
        weights=[embedding_matrix],
        trainable=m["trainable_embeddings"],
    )

    # Explicit Input so the model builds eagerly (Keras 3 is otherwise lazy and
    # has no output_shape until first call). maxlen fixes the sequence length.
    model = Sequential(name=f"sentiment_{arch}")
    model.add(Input(shape=(maxlen,), dtype="int32"))
    model.add(emb)

    if arch == "mlp":
        model.add(Flatten())
        model.add(Dense(m["dense_units"], activation="relu"))
        model.add(Dropout(m["dropout"]))
    elif arch == "pool":
        model.add(GlobalAveragePooling1D())
        model.add(Dense(m["dense_units"], activation="relu"))
        model.add(Dropout(m["dropout"]))
    elif arch == "lstm":
        model.add(LSTM(m["lstm_units"]))
        model.add(Dropout(m["dropout"]))
    else:
        raise ValueError(f"unknown architecture: {arch!r}")

    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
