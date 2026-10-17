
import numpy as np
from astropy import units as u
from scipy.spatial.transform import Rotation



def align(pos, axis, axis_ref=(0, 0, 1)):
    a1 = np.array(axis_ref, float)
    a2 = np.array(axis, float)

    a3 = np.cross(a1, a2)  # order matters: axis_ref x axis

    n1 = np.linalg.norm(a1)
    n2 = np.linalg.norm(a2)
    angle = -np.arccos(np.dot(a1, a2) / (n1 * n2))  # note the minus sign

    if angle == 0:
        return pos

    a3 = a3 / np.linalg.norm(a3) * angle
    R = Rotation.from_rotvec(a3)
    return R.apply(pos)


def project_to_pixels(sat_pos_kpc, header):
    """
    sat_pos_kpc : (N,3) array, satellite positions in kpc, ALREADY in the
                  same frame (same center, same units) as the particles
                  pNbody rotated to build this image.
    header      : astropy FITS header of the pNbody output image.

    Returns (col, row) array indices into the SAVED image array, i.e.
    image_data[row, col] is the pixel a satellite lands in. This matches
    pNbody's actual pixel-conversion path (ccd.py MapUsingGauss /
    MapUsingSpline -> mapping.c) including the np.rot90() applied to the
    final map -- NOT the naive linspace(xmin,xmax,nx) scaling, which is
    computed in instrument.py but never actually used for the mapping.
    """
    los = _parse_los_string(header["OBJ_LOS"])
    distance = header["OBJ_DIST"] * u.Mpc

    rotated = align(sat_pos_kpc, axis=los)

    x_kpc = rotated[:, 0] * u.kpc
    y_kpc = rotated[:, 1] * u.kpc

    x_arcsec = np.arctan(x_kpc / distance).to(u.arcsec).value
    y_arcsec = np.arctan(y_kpc / distance).to(u.arcsec).value

    # arcsec -> pixel factors, exactly as ccd.get_arcsec_to_pixel_factors()
    fx = 1.0 / header["PIXFOVX"]
    fy = 1.0 / header["PIXFOVY"]

    nx = header["NX"]
    ny = header["NY"]

    # pixel position BEFORE pNbody's final np.rot90() (see mapping.c:
    # ix = pos[:,0], iy = pos[:,1] used directly as array indices into
    # a (nx,ny) matrix)
    ix = x_arcsec * fx + nx / 2.0
    iy = y_arcsec * fy + ny / 2.0

    # np.rot90(raw)[row, col] == raw[col, ny_raw-1-row] (verified empirically
    # against numpy's actual rot90 semantics) -> invert to get where a given
    # (ix,iy) in raw lands in the rotated (saved) array:
    col = ix
    row = ny - 1 - iy

    return col, row

def _parse_los_string(s):
    # OBJ_LOS is stored as "[%5.3f,%5.3f,%5.3f]" -- see obs_object.py get_parameters()
    s = s.strip().strip("[]")
    return np.array([float(v) for v in s.split(",")])