"""Building 3-channel (multi-scaler) model inputs from a single image array.

Composes three already-fitted Scaler instances (see scalers.py) as the
three channels of a single input tensor, rather than picking one scaler.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from wdm_cnn.preprocessing.scalers import Scaler


def make_3channel_scaler_input(
    data: NDArray[np.floating],
    scaler_shl: Scaler,
    scaler_arcsinh: Scaler,
    scaler_std,
    chunk_size: Optional[int] = None,
) -> NDArray[np.float32]:
    """Apply three scalers to `data` and stack them as channels.

    Each (image, channel) pair is then independently standardized to
    zero mean, unit std -- so every image contributes comparably to
    training regardless of its own overall brightness.

    Args:
        data: Images, shape (N, H, W).
        scaler_shl: A fitted ShiftedLogScaler (or compatible) -> channel 0.
        scaler_arcsinh: A fitted ArcsinhScaler (or compatible) -> channel 1.
        scaler_std: A fitted scikit-learn StandardScaler (or anything with
            the same (n_samples, n_features)-shaped .transform interface)
            -> channel 2.
        chunk_size: If given, the per-image standardization step is done
            in batches of this many images at a time, rather than in one
            vectorized call. Since each image's normalization is fully
            independent of every other image's, this produces an
            identical result either way -- chunking only matters if you
            find it measurably reduces peak memory in your specific
            environment; the default (None = no chunking) is simpler and
            equally correct.

    Returns:
        Array of shape (N, 3, H, W), each (image, channel) slice having
        mean ~0 and std ~1.
    """
    ch_shl = scaler_shl.transform(data)
    ch_arc = scaler_arcsinh.transform(data)
    ch_std = scaler_std.transform(data.reshape(-1, 1)).reshape(data.shape)
    out = np.stack([ch_shl, ch_arc, ch_std], axis=1).astype(np.float32)

    step = chunk_size or out.shape[0]
    for i in range(0, out.shape[0], step):
        chunk = out[i : i + step]
        means = chunk.mean(axis=(2, 3), keepdims=True)
        stds = chunk.std(axis=(2, 3), keepdims=True) + 1e-8
        out[i : i + step] = (chunk - means) / stds

    return out