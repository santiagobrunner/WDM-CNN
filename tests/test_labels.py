import numpy as np
import pytest

from wdm_cnn.data.labels import (
    label_str2int,
)

DM_MODELS_3 = ["wdm07d5", "wdm10", "cdm"]
DM_MODELS_5 = ["wdm03d5", "wdm05", "wdm07d5", "wdm10", "cdm"]


def test_label_str2int_basic():
    labels = ["cdm", "wdm07d5", "wdm10", "cdm"]
    result = label_str2int(labels, DM_MODELS_3)
    np.testing.assert_array_equal(result, [2, 0, 1, 2])


def test_label_str2int_unknown_label_raises():
    with pytest.raises(ValueError):
        label_str2int(["not_a_model"], DM_MODELS_3)


