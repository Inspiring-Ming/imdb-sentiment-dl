#!/usr/bin/env python
"""
END-TO-END PIPELINE
===================
load IMDb -> leakage-free split -> train Word2Vec on TRAIN -> build embedding
matrix -> bake-off (mlp / pool / lstm) on the VALIDATION set -> retrain the
winner -> report on the held-out TEST set -> save model + plots.

    python run_pipeline.py            # full run
    python run_pipeline.py --quick    # small subset + 2 epochs (smoke test / CI)

Every stage prints what it did; artifacts land in models/ and reports/.
"""
from __future__ import annotations

import argparse
import json

from keras.callbacks import EarlyStopping

from src.config import load_config, resolve
from src.data import load_dataset
from src.embeddings import build_embedding_matrix, train_word2vec, tsne_coordinates
from src.evaluate import evaluate, plot_history, plot_tsne
from src.model import build_model


def _maybe_shrink(ds, quick: bool):
    """In --quick mode, subsample so the smoke test finishes in seconds."""
    if not quick:
        return ds
    n_tr, n_ev = 2000, 1000
    ds.X_train, ds.y_train = ds.X_train[:n_tr], ds.y_train[:n_tr]
    ds.X_val, ds.y_val = ds.X_val[:n_ev], ds.y_val[:n_ev]
    ds.X_test, ds.y_test = ds.X_test[:n_ev], ds.y_test[:n_ev]
    ds.tokens_train = ds.tokens_train[:n_tr]
    return ds


def main(quick: bool = False) -> None:
    cfg = load_config()
    epochs = 2 if quick else cfg["model"]["epochs"]
    batch = cfg["model"]["batch_size"]
    reports = lambda f: resolve(cfg, cfg["paths"]["reports_dir"], f)  # noqa: E731

    # ----- 1. DATA ------------------------------------------------------ #
    print("== 1. Load IMDb (leakage-free train/val/test) ==")
    ds = _maybe_shrink(load_dataset(cfg), quick)
    print(f"train={len(ds.X_train)}  val={len(ds.X_val)}  test={len(ds.X_test)}")

    # ----- 2. EMBEDDINGS ------------------------------------------------ #
    print("\n== 2. Train Word2Vec on TRAIN reviews ==")
    w2v = train_word2vec(cfg, ds.tokens_train)
    emb = build_embedding_matrix(cfg, w2v, ds.word_index)
    words, coords = tsne_coordinates(w2v)
    plot_tsne(words, coords, reports("tsne_embeddings.png"))

    # ----- 3. BAKE-OFF on the validation set ---------------------------- #
    print("\n== 3. Architecture bake-off (val accuracy) ==")
    es = EarlyStopping(patience=1, restore_best_weights=True, monitor="val_loss")
    scores = {}
    for arch in cfg["model"]["architectures"]:
        model = build_model(cfg, arch, emb)
        model.fit(ds.X_train, ds.y_train, validation_data=(ds.X_val, ds.y_val),
                  epochs=epochs, batch_size=batch, verbose=0, callbacks=[es])
        val_acc = model.evaluate(ds.X_val, ds.y_val, verbose=0)[1]
        scores[arch] = val_acc
        print(f"  {arch:5s}  val_acc = {val_acc:.4f}")
    best = max(scores, key=scores.get)
    print(f"\nWinner: {best}")

    # ----- 4. RETRAIN winner, REPORT on held-out TEST ------------------- #
    print(f"\n== 4. Retrain '{best}' and evaluate on TEST ==")
    model = build_model(cfg, best, emb)
    history = model.fit(ds.X_train, ds.y_train, validation_data=(ds.X_val, ds.y_val),
                        epochs=epochs, batch_size=batch, verbose=2)
    plot_history(history, reports("training_curves.png"))
    metrics = evaluate(model, ds.X_test, ds.y_test, out_png=reports("confusion_matrix.png"))

    # ----- 5. PERSIST --------------------------------------------------- #
    model_path = resolve(cfg, cfg["paths"]["models_dir"], f"sentiment_{best}.keras")
    model.save(model_path)
    print(f"\n[model] saved -> {model_path}")
    summary = {"best_arch": best, "val_scores": scores, "test_metrics": metrics}
    with open(reports("metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[report] metrics -> {reports('metrics.json')}")
    print("\nDone. Artifacts in models/ and reports/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke-test on a subset")
    main(**vars(ap.parse_args()))
