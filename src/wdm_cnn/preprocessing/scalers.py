"""Pixel-value normalization scalers for surface-brightness images.

All four scalers implement the schemes compared in the thesis (Section
4.1.1), following normalization approaches used elsewhere in astrophysics
(see Bernardini et al. 2021, 2025, as cited in the thesis). Each is
stateless (fitting is a no-op) and supports forward (`transform`) and
inverse (`inverse_transform`) mapping, so that model inputs/outputs can
be converted back to physical flux units when needed.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


class Scaler(ABC):
    """Common interface for all pixel-value scalers.

    Subclasses implement `transform` and `inverse_transform`. `fit` is a
    no-op for every scaler here (none require fitting to the data), kept
    only for interface compatibility with scikit-learn-style code (e.g.
    `StandardScaler`) that some scripts in this project use alongside
    these classes.
    """

    def fit(self, x: FloatArray) -> "Scaler":
        return self

    @abstractmethod
    def transform(self, x: FloatArray) -> FloatArray:
        """Map raw pixel values to normalized scale."""

    def fit_transform(self, x: FloatArray) -> FloatArray:
        return self.fit(x).transform(x)

    @abstractmethod
    def inverse_transform(self, x_scaled: FloatArray) -> FloatArray:
        """Map normalized values back to raw pixel values."""


class PowerLogScaler(Scaler):
    """Power-law + log normalization.

    x_norm = sign(x) * (1/k) * log10((|x|/x0)^q + c)

    Parameters k, x0, q, c are manually tuned (see thesis Section 4.1.1).
    """

    def __init__(self, k: float = 1.0, x0: float = 1e-17, q: float = 1.0, c: float = 1.0):
        self.k = k
        self.x0 = x0
        self.q = q
        self.c = c

    def transform(self, x: FloatArray) -> FloatArray:
        sign = np.sign(x)
        x_abs = np.abs(x)
        return sign * (1 / self.k) * np.log10((x_abs / self.x0) ** self.q + self.c)

    def inverse_transform(self, x_scaled: FloatArray) -> FloatArray:
        sign = np.sign(x_scaled)
        x_abs = np.abs(x_scaled)
        return sign * self.x0 * (10 ** (self.k * x_abs) - self.c) ** (1 / self.q)


class SinhLogScaler(Scaler):
    """Sinh-log normalization.

    x_norm = sign(x) * sinh(log(alpha * |x| + 1) / beta)

    Parameters alpha and beta are manually tuned (see thesis Section 4.1.1).
    Note: uses natural log (unlike PowerLogScaler and ShiftedLogScaler,
    which use log10) -- both directions below are self-consistent with
    this choice.
    """

    def __init__(self, alpha: float = 1e17, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta

    def transform(self, x: FloatArray) -> FloatArray:
        sign = np.sign(x)
        x_abs = np.abs(x)
        return sign * np.sinh(np.log(self.alpha * x_abs + 1) / self.beta)

    def inverse_transform(self, x_scaled: FloatArray) -> FloatArray:
        sign = np.sign(x_scaled)
        x_abs = np.abs(x_scaled)
        return sign * (np.exp(self.beta * np.arcsinh(x_abs)) - 1) / self.alpha


class ArcsinhScaler(Scaler):
    """Arcsinh (inverse hyperbolic sine) normalization.

    x_norm = arcsinh(x / soft)

    `soft` is the softening parameter setting the linear-to-log
    transition point (typically the per-pixel noise level, sigma_sky).
    """

    def __init__(self, soft: float = 1.0):
        self.soft = soft

    def transform(self, x: FloatArray) -> FloatArray:
        return np.arcsinh(x / self.soft)

    def inverse_transform(self, x_scaled: FloatArray) -> FloatArray:
        return self.soft * np.sinh(x_scaled)


class ShiftedLogScaler(Scaler):
    """Shifted-log normalization.

    x_norm = log10(x + shift)

    `shift` is typically set to 5*sigma_sky, ensuring the argument stays
    positive even for noise-dominated (near-zero or negative) pixels.
    """

    def __init__(self, shift: float):
        self.shift = shift

    def transform(self, x: FloatArray) -> FloatArray:
        return np.log10(x + self.shift)

    def inverse_transform(self, x_scaled: FloatArray) -> FloatArray:
        return 10**x_scaled - self.shift
