import numpy as np
import pytest

from wdm_cnn.data.labels import (
    inverse_transform,
    inverse_transform_sigma,
    label_inv_mass,
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


def test_label_inv_mass_linear_values():
    labels = ["cdm", "wdm10", "wdm07d5"]
    result = label_inv_mass(labels, DM_MODELS_3, transform="linear")
    np.testing.assert_allclose(result, [0.0, 0.1, 1 / 7.5], atol=1e-6)


def test_label_inv_mass_unknown_label_raises():
    with pytest.raises(ValueError, match="not present in dm_models"):
        label_inv_mass(["not_a_model"], DM_MODELS_3)


def test_label_inv_mass_unknown_transform_raises():
    with pytest.raises(ValueError, match="Unknown transform"):
        label_inv_mass(["cdm"], DM_MODELS_3, transform="sqrt")


def test_inverse_transform_unknown_transform_raises():
    """Regression test: originally silently returned None for bad input."""
    with pytest.raises(ValueError, match="Unknown transform"):
        inverse_transform(np.array([0.1]), transform="sqrt")


def test_linear_round_trip():
    labels = ["cdm", "wdm10", "wdm07d5", "wdm03d5", "wdm05"]
    targets = label_inv_mass(labels, DM_MODELS_5, transform="linear")
    recovered = inverse_transform(targets, transform="linear")
    np.testing.assert_allclose(recovered, targets)


def test_log_round_trip():
    labels = ["cdm", "wdm10", "wdm07d5", "wdm03d5", "wdm05"]
    targets = label_inv_mass(labels, DM_MODELS_5, transform="log", epsilon=0.01)
    recovered = inverse_transform(targets, transform="log", epsilon=0.01)
    expected_raw = np.array(
        [0.0, 1 / 10.0, 1 / 7.5, 1 / 3.5, 1 / 5.0], dtype=np.float32
    )
    np.testing.assert_allclose(recovered, expected_raw, atol=1e-5)


def test_inverse_transform_sigma_matches_hand_calc():
    mu_log = np.array([0.0, 1.0, 2.0])
    sigma_log = np.array([0.1, 0.2, 0.05])
    result = inverse_transform_sigma(mu_log, sigma_log)
    expected = np.exp(mu_log) * sigma_log
    np.testing.assert_allclose(result, expected)