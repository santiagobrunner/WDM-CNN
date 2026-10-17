"""Scaling relations that turn a satellite's stellar mass into a size.

The chain used to size the satellite masks (and the CSVF analysis):

    M_star --(inverted SHMR)--> M_vir --(Bryan & Norman 1998)--> R_vir
           --(Kravtsov 2013)--> R_1/2

All radii are PROPER (physical) kpc at the redshift z given. Only numpy
and astropy are needed.
"""

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

# FIREbox cosmology (Planck 2015), the same numbers halo2pnb.py uses.
FIREBOX_COSMOLOGY = FlatLambdaCDM(H0=67.74, Om0=0.3089)

# --- Parameters to double-check against the papers / thesis -----------------
# Power-law SHMR:  M_star = SHMR_MSTAR_NORM * (M_vir / SHMR_MVIR_PIVOT) ** alpha
# CHECK SHMR_MSTAR_NORM against Garrison-Kimmel et al. (2017): 3e9 means
# M_star / M_vir = 0.3 at M_vir = 1e10 Msun, ten times the peak of any
# abundance-matching relation (~0.03); 3e6 would be physically plausible.
SHMR_MSTAR_NORM = 3e9
SHMR_MVIR_PIVOT = 1e10
# CHECK against the coefficient of your CSVF chapter: R_1/2 = coeff * R_vir.
KRAVTSOV_COEFF = 0.015


def _check_positive_mass(mstar) -> None:
    if np.any(np.asarray(mstar) <= 0):
        raise ValueError("stellar masses must be positive")


def bryan_norman_delta_vir(z, cosmology=FIREBOX_COSMOLOGY):
    """Bryan & Norman (1998) virial overdensity relative to the critical density."""
    x = cosmology.Om(z) - 1.0
    return 18 * np.pi**2 + 82 * x - 39 * x**2


def rho_crit_kpc3(z, cosmology=FIREBOX_COSMOLOGY):
    """Critical density at redshift z, in Msun / kpc^3 (physical)."""
    return cosmology.critical_density(z).to(u.Msun / u.kpc**3).value


def invert_gk17_shmr(mstar_arr, alpha_samp, mstar_norm=SHMR_MSTAR_NORM, mvir_pivot=SHMR_MVIR_PIVOT):
    """Invert the power-law SHMR: M_vir = pivot * (M_star / norm) ** (1 / alpha).

    Args:
        mstar_arr: 1-D array of stellar masses [Msun].
        alpha_samp: Slope(s); scalar or 1-D array.

    Returns:
        Array of shape (len(mstar_arr), len(alpha_samp)), in Msun.
    """
    mstar_arr = np.asarray(mstar_arr, dtype=float)
    if mstar_arr.ndim != 1:
        raise ValueError(f"mstar_arr must be 1-D, got shape {mstar_arr.shape}")
    _check_positive_mass(mstar_arr)
    alpha_samp = np.atleast_1d(np.asarray(alpha_samp, dtype=float))
    return mvir_pivot * (mstar_arr[:, None] / mstar_norm) ** (1.0 / alpha_samp)


def mstar_to_mvir_GK17(mstar_msun, alpha=2.5, **shmr_kwargs):
    """Point estimate of M_vir [Msun] for one slope (default alpha = 2.5, which
    pairs with the maximum scatter sigma = 1.81 dex: the conservative,
    enclosing-radius choice, not the full sigma = 0.2-1.81 sweep used for the
    CSVF onset uncertainty)."""
    mstar = np.atleast_1d(np.asarray(mstar_msun, dtype=float))
    return invert_gk17_shmr(mstar, [alpha], **shmr_kwargs)[:, 0]


def mvir_to_rvir_kpc(mvir_msun, z, cosmology=FIREBOX_COSMOLOGY):
    """M_vir [Msun] -> R_vir [proper kpc], from M = (4/3) pi R^3 Delta_vir rho_crit."""
    delta_vir = bryan_norman_delta_vir(z, cosmology)
    rho_c = rho_crit_kpc3(z, cosmology)
    return (3 * np.asarray(mvir_msun, dtype=float) / (4 * np.pi * delta_vir * rho_c)) ** (1 / 3)


def rvir_to_rhalf_kravtsov(rvir_kpc, coeff=KRAVTSOV_COEFF):
    """R_vir -> half-stellar-mass radius R_1/2 (Kravtsov 2013): coeff * R_vir."""
    return coeff * np.asarray(rvir_kpc, dtype=float)


def mstar_to_rhalf_kpc(
    mstar_msun, z, alpha=2.5, cosmology=FIREBOX_COSMOLOGY,
    mstar_norm=SHMR_MSTAR_NORM, kravtsov_coeff=KRAVTSOV_COEFF,
):
    """The whole chain: stellar mass [Msun] -> R_1/2 [proper kpc] at redshift z."""
    mvir = mstar_to_mvir_GK17(mstar_msun, alpha, mstar_norm=mstar_norm)
    return rvir_to_rhalf_kravtsov(mvir_to_rvir_kpc(mvir, z, cosmology), kravtsov_coeff)