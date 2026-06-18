"""
EVALUATION + REPORTING
======================
The tutorial reports a single accuracy curve. A data scientist reports the full
picture: accuracy, precision/recall/F1, a confusion matrix, and the training
curve that reveals over-fitting. All plots are written to disk so the README and
the run are reproducible.

Use a non-interactive matplotlib backend so this runs headless (CI, servers).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASSES = ["negative", "positive"]


def evaluate(model, X, y_true, out_png: str | None = None) -> dict:
    proba = model.predict(X, verbose=0).ravel()
    y_pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"  accuracy = {acc:.4f}   f1 = {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    if out_png:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title("IMDb sentiment — confusion matrix")
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        plt.close()
        print(f"  [report] confusion matrix -> {out_png}")

    return {"accuracy": float(acc), "f1": float(f1)}


def plot_history(history, out_png: str) -> None:
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(h["accuracy"], label="train")
    ax1.plot(h["val_accuracy"], label="val")
    ax1.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax1.legend()
    ax2.plot(h["loss"], label="train")
    ax2.plot(h["val_loss"], label="val")
    ax2.set(title="Loss", xlabel="epoch", ylabel="loss")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  [report] training curves -> {out_png}")


def plot_tsne(words: list[str], coords: np.ndarray, out_png: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1])
    for w, (x, y) in zip(words, coords):
        plt.annotate(w, (x, y), xytext=(5, 2), textcoords="offset points")
    plt.title("Word2Vec embedding space (t-SNE) — sentiment words separate")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"  [report] t-SNE embeddings -> {out_png}")
