"""Round-trip tests: inverse_transform(transform(x)) should recover x.

These directly verify the algebra of each scaler's inverse -- a strong,
cheap correctness check given how easy it is to get sign handling and
operation order wrong when inverting nested log/sinh expressions.
"""

import numpy as np
import pytest

from wdm_cnn.preprocessing.scalers import (
    ArcsinhScaler,
    PowerLogScaler,
    ShiftedLogScaler,
    SinhLogScaler,
)

# A mix of positive, negative, small, and large values -- deliberately
# avoids exact zero, since PowerLogScaler's inverse involves a division
# by a signed zero that is mathematically fine but numerically awkward.
SAMPLE_VALUES = np.array([-1e2, -1e-3, 1e-5, 1e-3, 1.0, 1e2, 1e5])


def assert_round_trip(scaler, values=SAMPLE_VALUES, rtol=1e-6):
    scaled = scaler.transform(values)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, values, rtol=rtol)


def test_power_log_round_trip():
    assert_round_trip(PowerLogScaler(k=1.0, x0=1e-3, q=1.0, c=1.0))


def test_sinh_log_round_trip():
    assert_round_trip(SinhLogScaler(alpha=1e2, beta=1.0))


def test_arcsinh_round_trip():
    assert_round_trip(ArcsinhScaler(soft=1e-3))


def test_shifted_log_round_trip():
    # Only valid for x > -shift, since log10 of a non-positive number is
    # undefined; use only positive sample values for this one.
    positive_values = SAMPLE_VALUES[SAMPLE_VALUES > 0]
    scaler = ShiftedLogScaler(shift=1e-2)
    scaled = scaler.transform(positive_values)
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, positive_values, rtol=1e-6)


@pytest.mark.parametrize(
    "scaler",
    [
        PowerLogScaler(k=1.0, x0=1e-3, q=1.0, c=1.0),
        SinhLogScaler(alpha=1e2, beta=1.0),
        ArcsinhScaler(soft=1e-3),
    ],
)
def test_fit_transform_matches_transform(scaler):
    """fit() is a no-op; fit_transform should equal transform alone."""
    np.testing.assert_allclose(
        scaler.fit_transform(SAMPLE_VALUES), scaler.transform(SAMPLE_VALUES)
    )
