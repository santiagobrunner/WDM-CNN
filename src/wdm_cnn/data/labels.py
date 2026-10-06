"""Label encoding utilities.

`label_str2int` supports the core 3-class/5-class classification pipeline
(converting DM model names to integer class indices). The remaining
functions (`label_inv_mass`, `inverse_transform`, `inverse_transform_sigma`)
support the regression-based extension (predicting 1/m_WDM directly,
via RegrCNN) that goes beyond the thesis's classification scope.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from config import _INV_MASSES


def label_str2int(labels_str: Sequence[str], dm_models: Sequence[str]) -> NDArray[np.int64]:
    """Convert string DM-model labels to integer class indices.

    The index of each label within `dm_models` becomes its integer class.
    Raises ValueError (via list.index) if a label is not present in
    `dm_models`.
    """
    return np.array([dm_models.index(label) for label in labels_str], dtype=np.int64)


def label_inv_mass(
    labels_str: Sequence[str],
    dm_models: Sequence[str],
    transform: str = "linear",
    epsilon: float = 0.01,
) -> NDArray[np.float32]:
    """Convert string DM-model labels to inverse-mass regression targets.

    Args:
        labels_str: DM-model label for each sample.
        dm_models: Expected set of valid DM-model labels; used only to
            validate `labels_str` (every label must appear here).
        transform: 'linear' returns 1/m_WDM directly; 'log' returns
            log(1/m_WDM + epsilon), which keeps CDM's 1/m_WDM = 0 finite.
        epsilon: Offset added before taking the log, when transform='log'.

    Returns:
        Array of regression targets, one per label.

    Raises:
        ValueError: If any label in `labels_str` is not in `dm_models`,
            or if `transform` is not 'linear' or 'log'.
    """
    unknown = set(labels_str) - set(dm_models)
    if unknown:
        raise ValueError(f"Labels not present in dm_models: {unknown}")

    raw = np.array([_INV_MASSES[label] for label in labels_str], dtype=np.float32)

    if transform == "linear":
        return raw
    elif transform == "log":
        return np.log(raw + epsilon).astype(np.float32)
    else:
        raise ValueError(f"Unknown transform: {transform!r}. Expected 'linear' or 'log'.")


def inverse_transform(
    mu: NDArray[np.floating], transform: str = "linear", epsilon: float = 0.01
) -> NDArray[np.floating]:
    """Invert `label_inv_mass`'s target transform, mapping predictions
    back to physical inverse-mass (1/keV) units.

    Raises:
        ValueError: If `transform` is not 'linear' or 'log'.
    """
    if transform == "linear":
        return mu
    elif transform == "log":
        return np.exp(mu) - epsilon
    else:
        raise ValueError(f"Unknown transform: {transform!r}. Expected 'linear' or 'log'.")


def inverse_transform_sigma(
    mu_log: NDArray[np.floating], sigma_log: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Propagate an uncertainty from log-space to physical (1/m_WDM) space.

    Uses the delta method (first-order Taylor expansion of exp()):
    sigma_physical ~= exp(mu_log) * sigma_log. This is an approximation,
    accurate when sigma_log is small relative to the curvature of exp()
    -- i.e. for confidently-predicted (low-uncertainty) samples.
    """
    return np.exp(mu_log) * sigma_log