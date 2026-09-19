import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from wdm_cnn.data.io import get_data, get_label, load_images
from wdm_cnn.data.photometry import add_noise, get_flux

DM_MODELS = ["wdm07d5", "wdm10", "cdm"]


def _write_fits_gz(path: Path, data: np.ndarray, header: dict | None = None) -> None:
    plain_path = path.with_name(path.name[: -len(".gz")])
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    if header:
        for k, v in header.items():
            hdu.header[k] = v
    hdu.writeto(plain_path, overwrite=True)
    with open(plain_path, "rb") as f_in, gzip.open(path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    plain_path.unlink()


DEFAULT_HEADER = {
    "PIXFOVX": 16.11,
    "PIXAREA": 259.5,
    "UNITS": "flux",
    "NAXIS1": 4,
    "NAXIS2": 4,
}


@pytest.fixture
def halo_dir(tmp_path):
    halo = tmp_path / "FB15_h15_L11_r4_p1"
    snap_dir = halo / "100"
    snap_dir.mkdir(parents=True)

    for dm in DM_MODELS:
        filename = f"SB_h15_{dm}_100_LOS00_N32.fits.gz"
        _write_fits_gz(snap_dir / filename, np.full((4, 4), 1.0), DEFAULT_HEADER)

    (snap_dir / "readme.txt").write_text("not an image")
    return tmp_path, halo.name


# --- io.py tests ---


def test_get_label_matches():
    assert get_label("SB_h15_cdm_100_LOS00_N32.fits.gz", DM_MODELS) == "cdm"


def test_get_label_no_match_returns_none():
    assert get_label("SB_h15_unknown_100.fits.gz", DM_MODELS) is None


def test_load_images_basic(halo_dir):
    base_path, halo_name = halo_dir
    images, labels, files = load_images(str(base_path / halo_name), DM_MODELS, snap=100)
    assert images.shape == (3, 4, 4)
    assert sorted(labels) == sorted(DM_MODELS)
    assert len(files) == 3


def test_load_images_with_real_noise_injection(halo_dir):
    """Integration test: load_images -> add_noise (via the stub pNbody)."""
    base_path, halo_name = halo_dir
    images, labels, files = load_images(
        str(base_path / halo_name), DM_MODELS, snap=100, apply_noise=True, SB_limit=32
    )
    assert images.shape == (3, 4, 4)
    # noise was added: values should no longer be exactly the original 1.0
    assert not np.all(images == 1.0)
    # negative-value clipping guarantees strictly positive output
    assert np.all(images > 0)


def test_get_data_excludes_pairs(halo_dir):
    base_path, halo_name = halo_dir
    data, labels, files = get_data(
        str(base_path), halos=[halo_name], dm_models=DM_MODELS,
        snapshots=[100], exclude=[(15, "cdm")],
    )
    assert "cdm" not in labels
    assert len(data) == 2


def test_get_data_skips_missing_snapshot_without_error(halo_dir):
    base_path, halo_name = halo_dir
    data, labels, files = get_data(
        str(base_path), halos=[halo_name], dm_models=DM_MODELS, snapshots=[999]
    )
    assert len(data) == 0


# --- photometry.py tests ---


def test_add_noise_missing_header_key_raises():
    data = np.ones((4, 4))
    header = fits.Header()  # empty -- no PIXFOVX etc.
    with pytest.raises(ValueError, match="PIXFOVX"):
        add_noise(data, header, SB_limit=32)


def test_add_noise_no_sb_limit_no_sigma_raises():
    data = np.ones((4, 4))
    header = fits.Header(DEFAULT_HEADER)
    with pytest.raises(ValueError, match="SB_limit or sigma_per_pixel"):
        add_noise(data, header, SB_limit=None, sigma_per_pixel=None)

def test_add_noise_converts_magnitudes():
    data = np.full((4, 4), 20.0)
    header = fits.Header({**DEFAULT_HEADER, "UNITS": "mag/arcsec^2"})
    result = add_noise(data, header, SB_limit=32, zp=0)
    # Should be flux-like (much smaller than the raw magnitude value 20),
    # strictly positive after clipping.
    assert np.all(result > 0)
    assert np.all(result < 1.0)


def test_add_noise_clips_negative_to_positive_floor():
    data = np.full((4, 4), -1e10)  # deliberately very negative
    header = fits.Header(DEFAULT_HEADER)
    result = add_noise(data, header, SB_limit=32)
    assert np.all(result > 0)


def test_get_flux_matches_manual_calc():
    mag = np.array([20.0, 25.0])
    result = get_flux(mag, omega=2.0, zp=1.0)
    expected = 2.0 * 10 ** (-(mag - 1.0) / 2.5)
    np.testing.assert_allclose(result, expected)


def test_get_flux_default_matches_original_formula():
    """Defaults (omega=1, zp=0) should match the original get_flux(mag)."""
    mag = np.array([20.0, 25.0])
    result = get_flux(mag)
    expected = 10 ** (-mag / 2.5)
    np.testing.assert_allclose(result, expected)