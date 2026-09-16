"""Generation of lines-of-sight (LOS) vectors for mock imaging projections."""

import healpy as hp
import numpy as np
from numpy.typing import NDArray


def generate_lines_of_sight(nside: int = 2) -> NDArray[np.float64]:
    """Generate uniformly tiled unit LOS vectors via HEALPix.

    Selects HEALPix pixel centers on the upper hemisphere together with
    half of the equatorial ring (theta < pi/2, plus theta == pi/2 with
    phi < pi), giving 24 independent viewing directions for nside=2 -- as
    used throughout the thesis to avoid duplicate antipodal projections.

    The resulting vectors are rotated about the z-axis so that LOS index
    20 aligns exactly with the +x axis. This is an arbitrary but fixed
    reference-frame convention (does not affect relative LOS geometry).

    Args:
        nside: HEALPix resolution parameter. Default 2 gives 24 LOS,
            matching Table 11 (Appendix A) of the thesis.

    Returns:
        Array of shape (N, 3) of unit vectors; N = 24 for nside=2.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    mask = (theta < np.pi / 2) | ((theta == np.pi / 2) & (phi < np.pi))
    theta_sel, phi_sel = theta[mask], phi[mask]

    x = np.sin(theta_sel) * np.cos(phi_sel)
    y = np.sin(theta_sel) * np.sin(phi_sel)
    z = np.cos(theta_sel)
    los = np.column_stack([x, y, z])

    reference_index = 20
    delta_phi = np.arctan2(los[reference_index, 1], los[reference_index, 0])
    cos_a, sin_a = np.cos(-delta_phi), np.sin(-delta_phi)
    r_z = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return (r_z @ los.T).T
