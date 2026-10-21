"""Figures for the evaluation stage.

These functions only draw: they take the result dicts from metrics.py
(or the rows from aggregation.scan_n) and return a matplotlib Figure.
They never call plt.show() or save anything, so the calling script
decides:

    fig = plot_confusion_matrices(res, class_names)
    fig.savefig("cm.png", dpi=300, bbox_inches="tight")   # or plt.show()
    plt.close(fig)                                         # in loops

Matplotlib only (no seaborn).
"""

from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _heatmap(fig, ax, matrix, class_names: Sequence[str], fmt: str, vmin=0.0, vmax=None) -> None:
    """Annotated square heatmap; NaN cells are shown as 'n/a'."""
    matrix = np.asarray(matrix, dtype=float)
    n = len(class_names)
    if matrix.shape != (n, n):
        raise ValueError(f"matrix has shape {matrix.shape} but {n} class names were given")

    if vmax is None:
        vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    im = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names)

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if np.isnan(v):
                text, color = "n/a", "gray"
            else:
                text = str(int(v)) if fmt == "d" else format(v, fmt)
                color = "white" if v > 0.5 * vmax else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)


def plot_confusion_matrices(
    result: Mapping, class_names: Sequence[str], figsize=(12, 4), title: Optional[str] = None
):
    """Counts (left) and row-normalised (right) confusion matrices.

    Args:
        result: Dict from `evaluate_classification` / `evaluate_binary`
            (uses 'confusion_matrix' and 'confusion_matrix_normalized').
        class_names: Tick labels, in class-index order.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _heatmap(fig, axes[0], result["confusion_matrix"], class_names, "d")
    axes[0].set_title("Confusion Matrix (counts)")
    _heatmap(fig, axes[1], result["confusion_matrix_normalized"], class_names, ".2f", vmin=0.0, vmax=1.0)
    axes[1].set_title("Confusion Matrix (normalized)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confidence_histogram(
    result: Mapping,
    bins: int = 20,
    figsize=(6, 3),
    title: str = "Confidence Distribution",
    xlabel: str = "Max predicted probability",
):
    """Histogram of the top class probability, correct vs incorrect.

    Both histograms share the same bin edges (the old version let each
    pick its own, so the bars were not comparable).

    Args:
        result: Dict from `evaluate_classification` / `evaluate_binary`
            (uses 'max_prob' and 'correct').
    """
    max_prob = np.asarray(result["max_prob"], dtype=float)
    correct = np.asarray(result["correct"], dtype=bool)
    if max_prob.size == 0:
        raise ValueError("no predictions to plot")

    lo = float(max_prob.min())
    if lo >= 1.0:
        lo = 0.0
    edges = np.linspace(lo, 1.0, bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(max_prob[correct], bins=edges, alpha=0.5, label="Correct", color="green")
    ax.hist(max_prob[~correct], bins=edges, alpha=0.5, label="Incorrect", color="red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roc(result: Mapping, figsize=(5, 4), title: str = "ROC Curve"):
    """ROC curve from an `evaluate_binary` result (uses 'fpr', 'tpr', 'auc').

    Raises:
        ValueError: If the ROC curve is undefined (only one class present).
    """
    fpr, tpr = np.asarray(result["fpr"]), np.asarray(result["tpr"])
    if fpr.size == 0:
        raise ValueError("ROC curve is undefined: only one class present")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {result['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _as_columns(result) -> dict[str, np.ndarray]:
    """Columns of a scan result: a pandas DataFrame or a list of row dicts."""
    if hasattr(result, "columns"):
        return {c: np.asarray(result[c]) for c in result.columns}
    rows = list(result)
    if not rows:
        return {}
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in rows[0]}


def plot_n_scan(results, labels: Optional[Sequence[str]] = None, title: str = "N-scan", figsize=None):
    """Accuracy (and AUC, if present) versus N for one or more scans.

    Args:
        results: One scan or a list of scans. Each scan is the list of
            row dicts returned by `scan_n`, or a DataFrame built from it
            (needs columns 'N' and 'overall_acc'; 'auc' adds a second
            panel, as for binary scans).
        labels: Legend entries. Default: each scan's 'model_suffix'
            column if it has one, else 'scan 1', 'scan 2', ...
    """
    if hasattr(results, "columns") or (len(results) > 0 and isinstance(results[0], Mapping)):
        results = [results]
    if len(results) == 0:
        raise ValueError("no scan results to plot")

    cols = [_as_columns(r) for r in results]
    for i, c in enumerate(cols):
        if "N" not in c or "overall_acc" not in c:
            raise ValueError(f"scan {i + 1} needs 'N' and 'overall_acc' columns, has {sorted(c)}")

    if labels is None:
        labels = [str(c["model_suffix"][0]) if "model_suffix" in c else f"scan {i + 1}"
                  for i, c in enumerate(cols)]
    # Matplotlib silently hides legend entries whose label starts with an
    # underscore, and model suffixes always do (e.g. '_SimpleCNN_shl_...').
    labels = [str(lbl).lstrip("_") or "scan" for lbl in labels]

    has_auc = any("auc" in c for c in cols)
    fig, axes = plt.subplots(
        1, 2 if has_auc else 1, figsize=figsize or ((12, 4) if has_auc else (6, 4)), squeeze=False
    )
    axes = axes[0]

    for c, lbl in zip(cols, labels):
        axes[0].plot(c["N"], c["overall_acc"], marker="o", label=lbl)
        if has_auc and "auc" in c:
            axes[1].plot(c["N"], c["auc"], marker="s", label=lbl)

    axes[0].set_xlabel("N (images aggregated)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title} \u2014 Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if has_auc:
        axes[1].set_xlabel("N (images aggregated)")
        axes[1].set_ylabel("AUC-ROC")
        axes[1].set_title(f"{title} \u2014 AUC")
        axes[1].axhline(0.5, ls="--", color="grey", lw=1, label="random")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig