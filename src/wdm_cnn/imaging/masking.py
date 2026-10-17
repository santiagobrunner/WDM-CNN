import numpy as np
import pandas as pd
from astropy import units as u
from wdm_cnn.scaling_relations import mstar_to_mvir_GK17, mvir_to_rvir_kpc, rvir_to_rhalf_kravtsov
from wdm_cnn.ahf import read_satellites_by_radius, get_halo_center, get_scale_factor
from wdm_cnn.projection import project_to_pixels


def satellite_mask_radii_pixels(mstar_msun, z, distance_mpc, pixfov_arcsec,
                                 min_radius_pix=2.0):
    """
    Full chain: M_star -> Mvir (GK17) -> Rvir (Bryan&Norman) -> R_1/2
    (Kravtsov) -> mask radius (2*R_1/2) -> pixels, with a 1-pixel floor.
    """
    mvir = mstar_to_mvir_GK17(mstar_msun)
    rvir_kpc = mvir_to_rvir_kpc(mvir, z)
    rhalf_kpc = rvir_to_rhalf_kravtsov(rvir_kpc)
    mask_radius_kpc = 12 * rhalf_kpc  # mask radius = 2*R_1/2, per project convention

    # kpc -> arcsec -> pixels, same small-angle + pixel-scale convention
    # pNbody itself uses (see obs_object.py set_for_exposure / ccd.py)
    distance = distance_mpc * u.Mpc
    radius_arcsec = np.arctan((mask_radius_kpc * u.kpc) / distance).to(u.arcsec).value
    radius_pix = radius_arcsec / pixfov_arcsec
    return np.maximum(radius_pix, min_radius_pix)


def build_satellite_mask(image_shape, col, row, radii_pix):
    """
    Boolean union-of-circles mask, shape == image_shape (ny, nx),
    matching the (col,row) convention from project_to_pixels().
    """
    ny, nx = image_shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    mask = np.zeros(image_shape, dtype=bool)
    for c, r, rad in zip(col, row, radii_pix):
        mask |= (xx - c)**2 + (yy - r)**2 <= rad**2
    return mask


def mask_fill_with_noise_local(image_data, mask, col, row, radii_pix, sigma_sky, seed=None, annulus_width=3):
    """
    Fill each satellite's masked region with noise matched to the LOCAL
    background immediately surrounding it (not a single global value),
    since background brightness varies strongly with distance from the
    host galaxy center.
    """
    rng = np.random.default_rng(seed)
    masked_image = image_data.copy()
    ny, nx = image_data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    for c, r, rad in zip(col, row, radii_pix):
        inner = (xx - c)**2 + (yy - r)**2 <= rad**2
        outer = (xx - c)**2 + (yy - r)**2 <= (rad + annulus_width)**2
        annulus = outer & ~inner

        if annulus.sum() < 5:
            local_bg = np.percentile(image_data[~mask], 25)  # fallback if annulus too small/edge case
        else:
            local_bg = np.median(image_data[annulus])

        n_pix = int(inner.sum())
        masked_image[inner] = rng.normal(loc=local_bg, scale=sigma_sky, size=n_pix)

    return masked_image


def mask_fill_with_noise_local_soft(image_data, mask_bool_unused, col, row, radii_pix, sigma_sky,
                                     seed=None, annulus_width=3, feather_px=1.5):
    """
    Same as mask_fill_with_noise_local, but blends the fill smoothly into
    the real image over `feather_px` pixels at the mask boundary, instead
    of a hard cutoff -- avoids the ring artifact without inflating the
    effective masked area disproportionately for small (near-floor-radius)
    satellites.
    """
    rng = np.random.default_rng(seed)
    masked_image = image_data.copy().astype(float)
    ny, nx = image_data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    full_mask = np.zeros((ny, nx), dtype=bool)

    for c, r, rad in zip(col, row, radii_pix):
        dist = np.sqrt((xx - c)**2 + (yy - r)**2)
        inner = dist <= rad
        outer = dist <= (rad + annulus_width)
        annulus = outer & ~inner
        full_mask |= inner

        local_bg = np.median(image_data[annulus]) if annulus.sum() >= 5 else np.percentile(image_data, 25)
        noise_fill = rng.normal(loc=local_bg, scale=sigma_sky, size=(ny, nx))

        # feathered blend weight: 1.0 well inside, 0.0 well outside, smooth transition
        weight = np.clip((rad + feather_px - dist) / feather_px, 0, 1)
        masked_image = weight * noise_fill + (1 - weight) * masked_image

    n_masked = int(full_mask.sum())
    return masked_image, n_masked


def full_mask_for_image(hdr, ahf_path, host_id, n_star_min=7, min_radius_pix=1.0):
    """
    End-to-end: AHF satellites -> centered -> rotated/projected -> mask
    radii -> boolean mask, for one pNbody image + its AHF file.
    Returns (mask, col, row) or (None, None, None) if no satellites.
    """
    # sat_pos, mstar = read_ahf_satellites(ahf_path, host_id, n_star_min=n_star_min)
    sat_pos, mstar = read_satellites_by_radius(ahf_path, host_id, radius_ckpc=237.09, n_star_min=n_star_min)    
    if len(sat_pos) == 0:
        return None, None, None, None

    xc, yc, zc = get_halo_center(ahf_path, host_id)
    sat_pos_centered = sat_pos - np.array([xc, yc, zc])
    
    a = get_scale_factor(ahf_path)
    z = 1.0 / a - 1.0
    sat_pos_centered *= a  # convert comoving -> proper kpc, same as the particles  

    col, row = project_to_pixels(sat_pos_centered, hdr)

    radii_pix = satellite_mask_radii_pixels(
        mstar, z, hdr["OBJ_DIST"], hdr["PIXFOVX"], min_radius_pix=min_radius_pix)

    mask = build_satellite_mask((hdr["NY"], hdr["NX"]), col, row, radii_pix)
    return mask, col, row, radii_pix
