"""Classification metrics for the evaluation stage.

Everything here takes arrays of per-image class probabilities and true
labels, and returns numbers or arrays: nothing is printed or plotted
(plotting lives in plotting.py, and the calling script decides what to
print or save). Pure numpy plus a few sklearn metrics: no torch, no
pNbody.

Binary evaluation has two parts that must use different data: choosing
the decision threshold (`choose_threshold`, run on the VALIDATION split)
and applying it (`evaluate_binary(..., threshold=...)`, run on the TEST
split). Choosing it on the data you report on makes the numbers
optimistic.
"""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Candidate thresholds for P(heavy): 0.10, 0.11, ..., 0.90
DEFAULT_THRESHOLDS = np.linspace(0.1, 0.9, 81)


def _validate(probs, labels) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n_images, n_classes), got shape {probs.shape}")
    if len(labels) != len(probs):
        raise ValueError(f"probs and labels differ in length: {len(probs)} vs {len(labels)}")
    if labels.size == 0:
        raise ValueError("no images to evaluate")
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError(f"labels must lie in [0, {probs.shape[1] - 1}]")
    return probs, labels


def collapse_to_binary(
    probs: NDArray[np.floating],
    labels: Sequence[int],
    heavy_indices: Sequence[int] = (1, 2),
    light_indices: Sequence[int] = (0,),
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Merge classes of a multi-class model into a light/heavy binary problem.

    Probabilities of the merged classes are summed. The defaults,
    heavy=(1, 2) and light=(0,), correspond to the class order
    [wdm07d5, wdm10, cdm], i.e. 'wdm07d5 vs the rest' (thesis
    Boundary 0). Use light=(0, 1), heavy=(2,) for 'any WDM vs CDM'
    (Boundary 1).

    Returns:
        (bin_probs, bin_labels): shapes (M, 2) with column 0 = light and
        column 1 = heavy, and (M,) with 0 = light, 1 = heavy.

    Raises:
        ValueError: If the two index sets overlap or do not together
            cover every class exactly once.
    """
    probs, labels = _validate(probs, labels)
    n_classes = probs.shape[1]
    light, heavy = list(light_indices), list(heavy_indices)

    if set(light) & set(heavy):
        raise ValueError(f"light {light} and heavy {heavy} indices overlap")
    if set(light) | set(heavy) != set(range(n_classes)):
        raise ValueError(
            f"light {light} and heavy {heavy} must together cover every class "
            f"0..{n_classes - 1}"
        )

    bin_probs = np.stack([probs[:, light].sum(axis=1), probs[:, heavy].sum(axis=1)], axis=1)
    mapping = np.empty(n_classes, dtype=int)
    mapping[light] = 0
    mapping[heavy] = 1
    return bin_probs, mapping[labels]


def choose_threshold(
    scores: NDArray[np.floating],
    labels: Sequence[int],
    thresholds: Optional[Sequence[float]] = None,
) -> tuple[float, float]:
    """Pick the P(heavy) threshold that maximises balanced accuracy.

    Run this on VALIDATION data and pass the result to `evaluate_binary`
    / `scan_n` for the test data. Predict 'heavy' when score >= threshold.
    If several thresholds tie, the smallest is returned.

    Args:
        scores: P(heavy) per image, shape (M,).
        labels: Binary labels, 0 = light, 1 = heavy; both must occur.
        thresholds: Candidates; defaults to 0.10 ... 0.90 in steps of 0.01.

    Returns:
        (best_threshold, balanced_accuracy_at_that_threshold)
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise ValueError(f"scores and labels must be 1-D and equal length, got "
                         f"{scores.shape} and {labels.shape}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (light) or 1 (heavy)")
    heavy, light = labels == 1, labels == 0
    if not heavy.any() or not light.any():
        raise ValueError("both classes must be present to choose a threshold")

    cand = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)
    pred_heavy = scores[None, :] >= cand[:, None]  # (n_thresholds, M)
    tpr = pred_heavy[:, heavy].mean(axis=1)
    tnr = (~pred_heavy[:, light]).mean(axis=1)
    balanced = 0.5 * (tpr + tnr)
    best = int(np.argmax(balanced))
    return float(cand[best]), float(balanced[best])


def _summarize(labels, preds, class_names: Sequence[str]) -> dict:
    """Accuracy, balanced accuracy, confusion matrices and per-class report."""
    n = len(class_names)
    idx = list(range(n))
    cm = confusion_matrix(labels, preds, labels=idx)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.full(cm.shape, np.nan), where=row_sums > 0)
    recalls = np.diag(cm_norm)  # per-class recall; NaN for classes with no images
    return {
        "accuracy": float((labels == preds).mean()),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "report": classification_report(
            labels, preds, labels=idx, target_names=list(class_names),
            zero_division=0, output_dict=True,
        ),
    }


def get_summary(fits_path, pnbody_path, ahf_path):
    """
    Compile a full summary of a galaxy image.
    
    Args:
        fits_path   : path to the FITS file
        pnbody_path : path to the pNbody HDF5 file
        ahf_path    : path to the AHF halos file
    """
    halo_name = fits_path.split(sep='/')[5]
    dm_model = ahf_path.split(sep='/')[-1]
    dm_model = dm_model.split('.')[0]
    
    # --- FITS header info ---
    hdul=fits.open(fits_path)
    header = hdul[0].header

    # --- pNbody info ---
    nb = Nbody(pnbody_path)
    M_star = nb.mass.sum() *1e10
    N_star = len(nb.pos)
    mean_age = nb.age.mean()

    # --- AHF info ---
    df = pd.read_csv(ahf_path, sep=r'\s+')
    M_halo = df.iloc[0, 3]
    R_halo = df.iloc[0, 11]  # Rhalo column
    pos = df.iloc[0, 5:8].values
    vel = df.iloc[0, 8:11].values

    # --- Print summary ---
    print('='*60)
    print('GALAXY SUMMARY')
    print('='*60)
    
    print('\n--- Halo Information ---')
    # print(f'Halo name    : {header["OBJ_NAME"]}')
    print(f'DM model    : {dm_model}')
    print(f'Halo name    : {halo_name}')
    print(f'Halo ID    : {df.iloc[0, 0]}')
    # print('Halo Index: 0 (most massive halo)')
    print(f'Halo position: {pos} ckpc/h')
    print(f'Halo velocity: {vel} km/s')
    print(f'Halo mass    : {M_halo:.3e} Msun/h')
    print(f'Halo radius  : {R_halo:.2f} ckpc/h')
    
    print('\n--- Stellar Population ---')
    print(f'Stellar mass : {M_star:.3e} Msun')
    print(f'N stars      : {N_star}')
    print(f'Mean age     : {mean_age:.2f} Gyr')

    
    print('\n--- Instrument & Observation ---')
    print(f'Instrument   : {header["NAME"]}')
    print(f'Filter       : {header["FILTER"]}')
    print(f'Distance     : {header["OBJ_DIST"]} Mpc')
    print(f'Pixel scale  : {header["PIXFOVX"]:.3f} arcsec/pixel')
    print(f'FOV          : {header["FOVX"]:.3f} x {header["FOVY"]:.3f} deg')
    print(f'Image size   : {header["NX"]} x {header["NY"]} pixels')
    print(f'Line of sight: {header["OBJ_LOS"]}')
    
    print('\n--- pNbody Parameters ---')
    print(f'Smoothing    : {header["RSPMODE"]}')
    print(f'Smoothing fac: {header["RSPFAC"]}')
    print(f'Units        : {header["UNITS"]}')

    print('\n--- Command used ---')
    for comment in header['COMMENT']:
        print(comment)
    
    print('='*60)

    return header, nb, df

def evaluate_classification(
    probs: NDArray[np.floating], labels: Sequence[int], class_names: Sequence[str]
) -> dict:
    """Metrics for a multi-class model (prediction = most probable class).

    Returns a dict with: accuracy, balanced_accuracy, confusion_matrix
    (counts), confusion_matrix_normalized (rows sum to 1; NaN rows for
    classes with no images), report (sklearn's per-class dict),
    predictions, labels, probabilities, max_prob and correct (boolean),
    the last two for confidence histograms.
    """
    probs, labels = _validate(probs, labels)
    if probs.shape[1] != len(class_names):
        raise ValueError(f"probs has {probs.shape[1]} columns but {len(class_names)} class names")
    preds = probs.argmax(axis=1)
    out = _summarize(labels, preds, class_names)
    out.update(
        predictions=preds, labels=labels, probabilities=probs,
        max_prob=probs.max(axis=1), correct=preds == labels,
    )
    return out


def evaluate_binary(
    probs: NDArray[np.floating],
    labels: Sequence[int],
    heavy_indices: Sequence[int] = (1, 2),
    light_indices: Sequence[int] = (0,),
    class_names: Sequence[str] = ("light", "heavy"),
    threshold: Optional[float] = None,
) -> dict:
    """Metrics for a multi-class model collapsed to light vs heavy.

    Args:
        threshold: Predict 'heavy' when P(heavy) >= threshold. None means
            plain argmax (about 0.5). Choose it with `choose_threshold` on
            validation data, never on the data evaluated here.

    Returns:
        Everything `evaluate_classification` returns, for the collapsed
        problem, plus: threshold_used, auc (ROC-AUC of P(heavy); NaN if
        only one class is present), fpr and tpr (ROC curve points, empty
        if undefined), bin_probs and bin_labels.
    """
    bin_probs, bin_labels = collapse_to_binary(probs, labels, heavy_indices, light_indices)
    if len(class_names) != 2:
        raise ValueError("class_names must have exactly 2 entries for a binary evaluation")

    scores = bin_probs[:, 1]
    preds = bin_probs.argmax(axis=1) if threshold is None else (scores >= threshold).astype(int)

    out = _summarize(bin_labels, preds, class_names)
    both_present = len(np.unique(bin_labels)) == 2
    if both_present:
        fpr, tpr, _ = roc_curve(bin_labels, scores)
        auc = float(roc_auc_score(bin_labels, scores))
    else:
        fpr, tpr, auc = np.array([]), np.array([]), float("nan")

    out.update(
        threshold_used=0.5 if threshold is None else float(threshold),
        auc=auc, fpr=fpr, tpr=tpr,
        bin_probs=bin_probs, bin_labels=bin_labels, predictions=preds,
        max_prob=bin_probs.max(axis=1), correct=preds == bin_labels,
    )
    return out