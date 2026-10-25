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



def label_str2int(labels_str: Sequence[str], dm_models: Sequence[str]) -> NDArray[np.int64]:
    """Convert string DM-model labels to integer class indices.

    The index of each label within `dm_models` becomes its integer class.
    Raises ValueError (via list.index) if a label is not present in
    `dm_models`.
    """
    return np.array([dm_models.index(label) for label in labels_str], dtype=np.int64)

