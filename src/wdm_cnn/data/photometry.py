"""Noise injection and magnitude/flux conversions for mock images.

Built directly on pNbody.Mockimgs.noise -- importing this module requires
pNbody to be installed and importable.
"""

from typing import Optional

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from pNbody.Mockimgs import noise as pnbody_noise


def add_noise(
    data: NDArray[np.floating],
    header: fits.Header,
    SB_limit: Optional[float] = 31,
    SB_area: float = 100,
    SB_SN: float = 1,
    zp: float = 0,
    sigma_per_pixel: Optional[float] = None,
) -> NDArray[np.floating]:
    """Add Gaussian noise to a mock image, calibrated to a target depth.

    Args:
        data: Image data, either in mag/arcsec^2 (if header['UNITS'] says
            so) or already in flux units.
        header: FITS header; must contain PIXFOVX, NAXIS1, NAXIS2, PIXAREA,
            UNITS.
        SB_limit: Target 3-sigma surface brightness depth. If given,
            `sigma_per_pixel` is ignored and the noise std is derived
            from SB_limit via pNbody's noise model.
        SB_area: Aperture area (arcsec^2) defining the SB_limit depth.
        SB_SN: Signal-to-noise ratio defining the SB_limit depth.
        zp: Photometric zero point used for the mag -> flux conversion.
        sigma_per_pixel: Explicit per-pixel noise std, used only if
            `SB_limit` is None.

    Returns:
        Flux-unit image with Gaussian noise added; negative values are
        clipped to a small positive floor (1e-40) rather than left
        negative.

    Raises:
        ValueError: If a required header keyword is missing, or if
            neither `SB_limit` nor `sigma_per_pixel` is provided.
    """
    pixfovx = header.get("PIXFOVX")
    if pixfovx is None:
        raise ValueError("'PIXFOVX' not found in the fits header.")

    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        raise ValueError("'NAXIS1'/'NAXIS2' not found in the fits header.")

    omega = header.get("PIXAREA")
    if omega is None:
        raise ValueError("'PIXAREA' not found in the fits header.")

    units = header.get("UNITS")
    if units is None:
        raise ValueError("'UNITS' not found in the fits header.")

    if units == "mag/arcsec^2":
        flux = omega * 10 ** (-(data - zp) / 2.5)
    else:
        flux = data

    if SB_limit is not None:
        sigma = pnbody_noise.get_std_for_SB_limit(SB_limit, pixfovx, area=SB_area, sn=SB_SN)
    elif sigma_per_pixel is not None:
        sigma = sigma_per_pixel
    else:
        raise ValueError("SB_limit or sigma_per_pixel must be provided.")

    noise_map = np.random.normal(loc=0.0, scale=sigma, size=(naxis1, naxis2))
    flux = flux + noise_map
    return np.where(flux < 0, 1e-40, flux)


def get_flux(mag: NDArray[np.floating], omega: float = 1.0, zp: float = 0.0) -> NDArray[np.floating]:
    """Convert magnitudes to flux: omega * 10^(-(mag - zp) / 2.5).

    `omega` (pixel area) and `zp` (zero point) default to 1.0 and 0.0
    respectively; pass the actual header values for a physically
    meaningful flux, consistent with the conversion used in `add_noise`.
    """
    return omega * 10 ** (-(mag - zp) / 2.5)