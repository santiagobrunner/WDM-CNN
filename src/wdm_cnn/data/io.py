"""Reading mock images (FITS files) off disk into training-ready arrays.

Note: importing this module requires pNbody to be installed and
importable, since it imports `add_noise` from `photometry.py`, which
depends on pNbody.Mockimgs.noise -- even for code paths that never
apply noise.
"""

import os
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from wdm_cnn.data.photometry import add_noise
from config import DEFAULT_SNAPSHOTS


def get_label(filename: str, dm_models: Sequence[str]) -> Optional[str]:
    """Identify which DM model a filename belongs to.

    Checks each entry in `dm_models` as a substring of `filename`, in
    order, and returns the first match. Returns None if no DM model
    name appears in the filename.

    Note: matching is order-dependent if any model name in `dm_models`
    is itself a substring of another (not the case for the standard
    3- or 5-class model lists used in this project).
    """
    for dm in dm_models:
        if dm in filename:
            return dm
    return None


def load_images(
    halo_path: str,
    dm_models: Sequence[str],
    snap: int,
    apply_noise: bool = False,
    SB_limit: float = 32,
    SB_area: float = 100,
    SB_SN: float = 1,
    zp: float = 0,
) -> tuple[NDArray[np.float32], list[str], list[str]]:
    """Load all mock images for one halo at one snapshot.

    Args:
        halo_path: Path to the halo directory (.../input_images/FB15_h*).
        dm_models: Recognized DM-model labels (used to identify each file).
        snap: Snapshot number; subdirectory is assumed to be `snap:03d`.
        apply_noise: If True, calls `add_noise` on each image using the
            SB_limit/SB_area/SB_SN/zp parameters below.
        SB_limit, SB_area, SB_SN, zp: Forwarded to `add_noise` when
            apply_noise=True; see its docstring.

    Returns:
        (images, labels, files): images is an (N, H, W) float32 array;
        labels and files are lists of length N. Files that aren't
        '.fits.gz', that fail to open, or whose filename doesn't match
        any entry in `dm_models` are silently skipped (a warning is
        printed for corrupt files only).
    """
    images: list[NDArray[np.float32]] = []
    labels: list[str] = []
    files: list[str] = []

    snap_path = os.path.join(halo_path, f"{snap:03d}")

    for filename in sorted(os.listdir(snap_path)):
        if not filename.endswith(".fits.gz"):
            continue

        path = os.path.join(snap_path, filename)
        try:
            with fits.open(path, ignore_missing_simple=True) as hdul:
                data = hdul[0].data.copy().astype(np.float32)
                header = hdul[0].header.copy()
                if apply_noise:
                    data = add_noise(data, header, SB_limit, SB_area, SB_SN, zp)
        except Exception as e:
            print(f"Warning: skipping corrupt file {path}: {e}")
            continue

        label = get_label(filename, dm_models)
        if label is None:
            continue

        images.append(data)
        labels.append(label)
        files.append(filename)

    return np.array(images, dtype=np.float32), labels, files


def get_data(
    input_path: str,
    halos: Sequence[str],
    dm_models: Sequence[str],
    snapshots: Sequence[int] = DEFAULT_SNAPSHOTS,
    apply_noise: bool = False,
    SB_limit: float = 32,
    SB_area: float = 100,
    SB_SN: float = 1,
    zp: float = 0,
    exclude: Sequence[tuple[int, str]] = (),
) -> tuple[NDArray[np.float32], NDArray[np.str_], NDArray[np.str_]]:
    """Load mock images across multiple halos and snapshots into one dataset.

    Args:
        input_path: Base path to the directory holding halo subdirectories.
        halos: Halo directory names (e.g. "FB15_h15_L11_r4_p1"). The halo
            number is parsed from the second underscore-separated field
            (e.g. "h15" -> 15) -- this assumes that naming convention.
        dm_models: Recognized DM-model labels.
        snapshots: Snapshot numbers to include.
        apply_noise, SB_limit, SB_area, SB_SN, zp: Forwarded to
            `load_images` -- see its docstring.
        exclude: (halo_number, dm_model) pairs to exclude from the result.

    Returns:
        (data, labels, files) as numpy arrays, one entry per image,
        concatenated across all halos and snapshots.
    """
    final_data: list[NDArray[np.float32]] = []
    final_labels: list[str] = []
    final_files: list[str] = []
    exclude_set = set(exclude)

    for halo in halos:
        halo_nr = int(halo.split("_")[1].replace("h", ""))
        halo_path = os.path.join(input_path, halo)

        for snap in snapshots:
            snap_path = os.path.join(halo_path, f"{snap:03d}")
            if not os.path.exists(snap_path):
                print(f"Skipping missing snap: {halo} snap{snap:03d}")
                continue

            images, labels, files = load_images(
                halo_path, dm_models, snap, apply_noise, SB_limit, SB_area, SB_SN, zp
            )
            for img, label, file in zip(images, labels, files):
                if (halo_nr, label) in exclude_set:
                    continue
                final_data.append(img)
                final_labels.append(label)
                final_files.append(file)

    return (
        np.array(final_data, dtype=np.float32),
        np.array(final_labels),
        np.array(final_files),
    )
