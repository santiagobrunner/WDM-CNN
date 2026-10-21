"""N-image aggregation: combining per-image class probabilities from
several independent galaxy images into one joint prediction.

Implements the rule of thesis Section 2.4.6 (Eq. 55): assuming the
images are conditionally independent given the class, the joint score of
class c is the sum of the per-image log-probabilities. "Independent"
here means the images come from distinct (halo, snapshot) groups, so no
two images of the same system (different lines of sight) are combined.

Pure numpy (plus sklearn's roc_auc_score for the binary case): no torch,
no pNbody. Sampling uses an explicit random Generator, so results are
reproducible given a seed.
"""

import os
from typing import Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

ClassGroups = dict[int, dict[tuple[str, int], list[int]]]


def extract_metadata(filepath: str) -> tuple[str, str, int]:
    """Parse a mock-image filename into (halo, dm_model, snapshot).

    Example: 'SB_h15_cdm_100_LOS00_N35.fits.gz' -> ('h15', 'cdm', 100).
    Assumes underscore-separated fields in exactly this order (the same
    filename convention the SLURM drivers and get_label rely on).
    """
    parts = os.path.basename(filepath).split("_")
    return parts[1], parts[2], int(parts[3])


def halos_and_snaps(files: Sequence[str]) -> tuple[NDArray, NDArray]:
    """Halo names and snapshot numbers for a list of image filenames."""
    meta = [extract_metadata(f) for f in files]
    return np.array([m[0] for m in meta]), np.array([m[2] for m in meta])


def build_class_groups(
    labels: Sequence[int], halos: Sequence[str], snaps: Sequence[int]
) -> ClassGroups:
    """Group image indices by class, then by (halo, snapshot).

    Returns {class: {(halo, snap): [image indices]}}. Images in one
    (halo, snap) group differ only by line of sight, so they are not
    independent and must never be combined in one aggregated draw.
    """
    if not (len(labels) == len(halos) == len(snaps)):
        raise ValueError(
            f"labels, halos and snaps must have equal length, got "
            f"{len(labels)}, {len(halos)}, {len(snaps)}"
        )
    groups: ClassGroups = {}
    for i, (label, halo, snap) in enumerate(zip(labels, halos, snaps)):
        groups.setdefault(int(label), {}).setdefault((str(halo), int(snap)), []).append(i)
    return groups


def sample_n_images_for_class(
    N: int, true_class: int, class_groups: ClassGroups, rng: np.random.Generator
) -> Optional[list[int]]:
    """Draw N images of one class, each from a different (halo, snap) group.

    Returns image indices, or None if the class has fewer than N groups.
    """
    group_map = class_groups.get(true_class, {})
    keys = list(group_map.keys())
    if len(keys) < N:
        return None
    chosen = rng.choice(len(keys), size=N, replace=False)
    return [int(rng.choice(group_map[keys[g]])) for g in chosen]


def aggregate_posteriors(
    N: int,
    probs: NDArray[np.floating],
    class_groups: ClassGroups,
    n_classes: int,
    n_trials: int = 100,
    rng: Optional[np.random.Generator] = None,
    model_prior: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Monte-Carlo draws of N-image aggregated class probabilities.

    In each of `n_trials` trials, one draw is made per class: N images
    from distinct (halo, snap) groups of that class, combined by summing
    log-probabilities and renormalising. Classes with fewer than N
    groups are skipped.

    Args:
        probs: Per-image class probabilities, shape (M, n_classes).
        model_prior: Class prior implicit in `probs`. None means
            uniform (balanced training, as for the 3-class model), where
            no correction is needed. For a model whose outputs were
            collapsed to fewer classes (e.g. binary from 3-class, whose
            implicit prior is (1/3, 2/3)), pass that prior: N*log(prior)
            is subtracted so the score is a likelihood ratio, the right
            rule when classes are evaluated with equal frequency (as
            here). Without it the sum of log-probabilities drifts
            toward the over-represented class as N grows.
        eps: Added inside the log to avoid log(0).

    Returns:
        (true_classes, agg_probs): shapes (T,) and (T, n_classes), where
        T is the number of draws actually made.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    probs = np.asarray(probs, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    log_prior = None if model_prior is None else np.log(np.asarray(model_prior, dtype=float))

    true_classes: list[int] = []
    agg: list[NDArray[np.float64]] = []
    for _ in range(n_trials):
        for c in range(n_classes):
            idx = sample_n_images_for_class(N, c, class_groups, rng)
            if idx is None:
                continue
            log_post = np.log(probs[idx] + eps).sum(axis=0)
            if log_prior is not None:
                log_post = log_post - N * log_prior
            p = np.exp(log_post - log_post.max())
            agg.append(p / p.sum())
            true_classes.append(c)

    return np.array(true_classes, dtype=int), np.array(agg).reshape(-1, n_classes)


def evaluate_at_n(
    N: int,
    probs: NDArray[np.floating],
    class_groups: ClassGroups,
    n_classes: int,
    n_trials: int = 100,
    rng: Optional[np.random.Generator] = None,
    model_prior: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
) -> dict[str, float]:
    """Accuracy of N-image aggregation, overall and per class.

    Args:
        threshold: Only for 2-class problems. Predict class 1 when its
            aggregated probability is >= threshold (instead of argmax).
            Choose it on validation data, not on the data being
            evaluated. Use either `threshold` (empirical) or
            `model_prior` (analytic), normally not both.

    Returns:
        Dict with 'overall_acc', 'mean_confidence' (mean of the largest
        aggregated probability), and 'class_{c}_acc' per class. For two
        classes it also contains 'auc' and 'threshold_used'. Entries are
        NaN where no draws could be made (N exceeds a class's number of
        groups).
    """
    if threshold is not None and n_classes != 2:
        raise ValueError("threshold is only defined for 2-class problems")

    truths, agg = aggregate_posteriors(N, probs, class_groups, n_classes, n_trials, rng, model_prior)

    nan = float("nan")
    out: dict[str, float] = {}
    if truths.size == 0:
        out["overall_acc"] = nan
        out["mean_confidence"] = nan
        for c in range(n_classes):
            out[f"class_{c}_acc"] = nan
        correct = None
    else:
        preds = agg.argmax(axis=1) if threshold is None else (agg[:, 1] >= threshold).astype(int)
        correct = preds == truths
        out["overall_acc"] = float(correct.mean())
        out["mean_confidence"] = float(agg.max(axis=1).mean())
        for c in range(n_classes):
            m = truths == c
            out[f"class_{c}_acc"] = float(correct[m].mean()) if m.any() else nan

    if n_classes == 2:
        out["threshold_used"] = 0.5 if threshold is None else float(threshold)
        both_present = truths.size > 0 and len(np.unique(truths)) == 2
        out["auc"] = float(roc_auc_score(truths, agg[:, 1])) if both_present else nan

    return out


def scan_n(
    probs: NDArray[np.floating],
    labels: Sequence[int],
    halos: Sequence[str],
    snaps: Sequence[int],
    class_names: Sequence[str],
    N_values: Sequence[int],
    n_trials: int = 100,
    seed: int = 0,
    model_prior: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> list[dict]:
    """Run `evaluate_at_n` for a range of N and return one row per N.

    N values larger than the smallest class's number of (halo, snap)
    groups are dropped. A single seeded Generator drives the whole scan,
    so one `seed` reproduces every row. Rows are plain dicts, ready for
    pd.DataFrame(rows).

    Args:
        class_names: Names for the columns 'class_{name}_acc'; its length
            sets the number of classes.
        metadata: Extra columns copied unchanged into every row, e.g.
            {'SB': 35, 'band': 'VIS', 'split': 'v1', 'model_suffix': s}.
            The aggregation itself doesn't use them; they keep each row
            self-describing when scans are concatenated. Keys that clash
            with result columns raise ValueError.
    """
    metadata = dict(metadata or {})
    probs = np.asarray(probs, dtype=float)
    n_classes = len(class_names)
    if probs.shape != (len(labels), n_classes):
        raise ValueError(
            f"probs has shape {probs.shape}, expected ({len(labels)}, {n_classes})"
        )

    class_groups = build_class_groups(labels, halos, snaps)
    max_N = min(len(class_groups.get(c, {})) for c in range(n_classes))
    valid_N = sorted({int(n) for n in N_values if 1 <= n <= max_N})

    rng = np.random.default_rng(seed)
    rows = []
    for N in valid_N:
        m = evaluate_at_n(N, probs, class_groups, n_classes, n_trials, rng, model_prior, threshold)
        result = {"N": N, "overall_acc": m["overall_acc"], "mean_confidence": m["mean_confidence"]}
        for c, name in enumerate(class_names):
            result[f"class_{name}_acc"] = m[f"class_{c}_acc"]
        if n_classes == 2:
            result["auc"] = m["auc"]
            result["threshold_used"] = m["threshold_used"]
        clash = metadata.keys() & result.keys()
        if clash:
            raise ValueError(f"metadata keys clash with result columns: {sorted(clash)}")
        rows.append({**metadata, **result})
    return rows