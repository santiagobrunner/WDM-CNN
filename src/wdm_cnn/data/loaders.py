"""Building PyTorch DataLoaders from saved mock images and trained scalers.

Paths that were previously hardcoded (cluster storage roots) are now
function parameters with defaults matching their original values -- this
makes them config-file-ready (e.g. driven by a YAML config later) without
introducing a config system before it's needed.
"""

import os
import pickle
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from wdm_cnn.data.io import get_data
from wdm_cnn.data.labels import label_str2int
from wdm_cnn.preprocessing.multiband import make_3channel_scaler_input
from config import CSCS_ROOT, CSCS_THESIS, UZH_ROOT, UZH_THESIS, DM_MODELS_3, DM_MODELS_5

if TYPE_CHECKING:
    # Only needed for type hints -- kept out of the runtime import path so
    # this module (and its pure suffix-building functions in particular)
    # doesn't require torch to be installed just to be imported/tested.
    from torch.utils.data import DataLoader

# (halo_number, dm_model) pairs excluded from training due to known data
# issues. Previously duplicated independently in three separate places
# (train_classifier.py, create_data_loader, create_multiband_test_loader);
# consolidated here as the single source of truth. Candidate to move into
# a config file if this needs to vary per-experiment in the future.
EXCLUDED_PAIRS: tuple[tuple[int, str], ...] = (
    (15, "wdm03d5"), (25, "wdm03d5"), (32, "wdm03d5"), (34, "wdm03d5"),
    (38, "wdm03d5"), (42, "wdm03d5"), (48, "wdm03d5"), (49, "wdm03d5"),
    (51, "wdm03d5"),
    (20, "wdm05"), (32, "wdm05"), (42, "wdm05"), (46, "wdm05"),
    (48, "wdm05"), (51, "wdm05"),
)

# NOTE: ConfigCNN is a thesis-relevant model but isn't wired into the
# suffix convention below yet -- the original training scripts never
# built a saved-scaler naming scheme for it (likely because hyperparameter
# search doesn't go through this same save/load path). Only models with
# an established suffix convention are listed here; add "ConfigCNN" once
# its naming convention is decided.
_SUPPORTED_MODELS = ("SimpleCNN", "CNN5_CLS")

_ROT_TAG_MAP = {"random180": "", "rot90": "_rot90", "none": "_norot"}


def dm_models_for(n_classes: int) -> list[str]:
    """Return the DM-model label list for a given number of classes."""
    if n_classes == 3:
        return list(DM_MODELS_3)
    elif n_classes == 5:
        return list(DM_MODELS_5)
    else:
        raise ValueError(f"Unknown n_classes: {n_classes}. Expected 3 or 5.")


def _build_classification_suffix(
    model: str, scaler_str: str, SB_limit: int, snapshots: str, n_classes: int,
    augment: bool, apply_exclusions: bool, online: bool, split: str,
    filter_str: str, rot_tag: str, masked: bool, res_suffix: str,
) -> str:
    """Build the filename suffix identifying one classification run.

    Pure string construction -- no I/O -- kept separate so it can be
    tested and reused without needing real saved files on disk.
    """
    suffix = f"_{model}_{scaler_str}_SBL{SB_limit}_z{snapshots}_n{n_classes}"
    suffix += "_aug" if augment else ""
    suffix += "_excl" if apply_exclusions else ""
    suffix += "_online" if online else ""
    suffix += f"_{split}" if split != "v1" else ""
    suffix += filter_str
    suffix += rot_tag
    suffix += "_masked" if masked else ""
    suffix += res_suffix
    return suffix


def load_split_arrays(
    data_type: str = "test",
    SB_limit: int = 35,
    snapshots: str = "all",
    n_classes: int = 3,
    scaler_str: str = "shl",
    augment: bool = True,
    model: str = "SimpleCNN",
    thesis_root: Optional[str] = None,
    images_root: Optional[str] = None,
    apply_exclusions: bool = True,
    split: str = "v1",
    online: bool = False,
    rotation: str = "rot90",
    band: str = "VIS",
    masked: bool = False,
    res_factor: int = 1,
    suffix: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load one halo split's images, scaled with the scaler saved at training time.

    This is the torch-free core of `create_data_loader`. Unlike the loader
    it also returns the image filenames, which the N-image aggregation
    needs to know each image's (halo, snapshot).

    Args:
        data_type: 'train', 'val', or 'test'.
        thesis_root: Base path holding halo split files, saved scalers,
            and post-stretch stats. Defaults to the CSCS cluster path.
            NOTE: training (train.py) uses the UZH paths, so pass the
            roots explicitly rather than relying on this default.
        images_root: Parent directory of the `input_images*` folders.
        apply_exclusions: If True, drop the (halo, dm) pairs in
            EXCLUDED_PAIRS from the result.
        masked, res_factor: Select the satellite-masked and/or
            resolution-degraded dataset variant, matching the suffix
            convention used when the corresponding model was trained.
        suffix: Use this model suffix as is, instead of building it from
            the arguments (for models without a suffix convention yet,
            e.g. ConfigCNN, or multi-band runs).

    Returns:
        (data_norm, targets, files, suffix): scaled images (N, H, W) or
        (N, 3, H, W) for 'AllSc'; integer class labels (N,); filenames
        (N,), in the same order as the images; and the suffix used.

    Raises:
        ValueError: For an unrecognized `model`, `data_type`, `snapshots`,
            `rotation` or `n_classes`.
    """
    thesis_root = thesis_root or CSCS_THESIS
    images_root = images_root or CSCS_ROOT

    if data_type not in ("train", "val", "test"):
        raise ValueError(f"Unknown data_type: {data_type!r}")

    if rotation not in _ROT_TAG_MAP:
        raise ValueError(f"Unknown rotation: {rotation!r}. Expected one of {sorted(_ROT_TAG_MAP)}.")

    dm_models = dm_models_for(n_classes)
    excluded = list(EXCLUDED_PAIRS) if apply_exclusions else []

    if snapshots == "all":
        snaps = [60, 63, 66, 69, 72, 75, 80, 85, 90, 95, 100]
    elif snapshots == "100":
        snaps = [100]
    else:
        raise ValueError(f"Unknown snapshots: {snapshots!r}")

    filter_str = f"_{band}" if band != "VIS" else ""
    mask_suffix = "_masked" if masked else ""
    res_suffix = f"_res{res_factor}x" if res_factor != 1 else ""
    rot_tag = _ROT_TAG_MAP[rotation]

    input_path = f"{images_root}/input_images{filter_str}_{SB_limit}_f{mask_suffix}{res_suffix}/"

    if suffix is None:
        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown or unsupported model: {model!r}. Expected one of {_SUPPORTED_MODELS}, "
                f"or pass an explicit `suffix`."
            )
        suffix = _build_classification_suffix(
            model, scaler_str, SB_limit, snapshots, n_classes, augment,
            apply_exclusions, online, split, filter_str, rot_tag, masked, res_suffix,
        )

    split_suffix = f"_{split}" if split != "v1" else ""
    halos = np.load(f"{thesis_root}/{data_type}_halos{split_suffix}.npy")

    data, labels, files = get_data(
        input_path, halos, dm_models, snaps, apply_noise=False, exclude=excluded
    )

    targets = label_str2int(labels, dm_models)

    if scaler_str == "AllSc":
        with open(f"{thesis_root}/Scalers/scalers{suffix}.pkl", "rb") as f:
            scalers = pickle.load(f)
        data_norm = make_3channel_scaler_input(data, scalers["shl"], scalers["as"], scalers["std"])
    else:
        with open(f"{thesis_root}/Scalers/scaler{suffix}.pkl", "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            scaler = loaded["scaler"]
            data_norm = scaler.transform(data)
            data_norm = (data_norm - loaded["post_mean"]) / loaded["post_std"]
        else:
            scaler = loaded
            if scaler_str == "std":
                data_norm = scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
            else:
                post_mean, post_std = np.load(f"{thesis_root}/Losses/post_stretch_stats{suffix}.npy")
                data_norm = scaler.transform(data)
                data_norm = (data_norm - post_mean) / post_std

    return data_norm, targets, np.asarray(files), suffix


def create_data_loader(
    data_type: str = "test",
    SB_limit: int = 35,
    snapshots: str = "all",
    n_classes: int = 3,
    scaler_str: str = "shl",
    augment: bool = True,
    model: str = "SimpleCNN",
    batch_size: int = 32,
    thesis_root: Optional[str] = None,
    images_root: Optional[str] = None,
    apply_exclusions: bool = True,
    split: str = "v1",
    online: bool = False,
    rotation: str = "rot90",
    band: str = "VIS",
    masked: bool = False,
    res_factor: int = 1,
    suffix: Optional[str] = None,
) -> "DataLoader":
    """Build a DataLoader for train/val/test, using a previously saved scaler.

    Arguments are those of `load_split_arrays` plus `batch_size`. Only the
    'train' loader shuffles and augments; for val/test the batch order
    matches the file order. (To get the matching filenames, use
    `load_split_arrays` or `wdm_cnn.evaluation.inference`.)
    """
    from torch.utils.data import DataLoader

    from wdm_cnn.models.models import GalaxyDataset

    data_norm, targets, _, _ = load_split_arrays(
        data_type=data_type, SB_limit=SB_limit, snapshots=snapshots, n_classes=n_classes,
        scaler_str=scaler_str, augment=augment, model=model, thesis_root=thesis_root,
        images_root=images_root, apply_exclusions=apply_exclusions, split=split,
        online=online, rotation=rotation, band=band, masked=masked,
        res_factor=res_factor, suffix=suffix,
    )

    use_augment = data_type == "train"
    dataset = GalaxyDataset(data_norm, targets, augmentation=use_augment, rotation=rotation)
    return DataLoader(dataset, batch_size=batch_size, shuffle=use_augment)


def create_multiband_test_loader(
    sb: int,
    n_classes: int,
    split: str,
    bands: Sequence[str],
    batch_size: int = 32,
    thesis_root: Optional[str] = None,
    images_root: Optional[str] = None,
) -> tuple["DataLoader", np.ndarray, str]:
    """Build a test DataLoader for a multi-band model.

    Args:
        thesis_root: Base path holding test-halo split files and saved
            scalers. Defaults to the original UZH cluster path.
        images_root: Parent directory of the per-band `input_images*`
            folders. Defaults to the original UZH cluster path.

    Returns:
        (test_loader, test_files, suffix): test_loader yields
        (x_multi, y) with x_multi of shape (B, n_bands, H, W); test_files
        are the reference-band filenames (for metadata extraction);
        suffix identifies the corresponding trained model.
    """
    from torch.utils.data import DataLoader

    from wdm_cnn.models.models import GalaxyDataset

    thesis_root = thesis_root or UZH_THESIS
    images_root = images_root or UZH_ROOT

    dm_models = dm_models_for(n_classes)
    snaps = [60, 63, 66, 69, 72, 75, 80, 85, 90, 95, 100]
    excluded = list(EXCLUDED_PAIRS) if n_classes == 5 else []

    bands_str = "".join(bands)
    suffix = f"_SimpleCNN_shl_SBL{sb}_zall_n{n_classes}_multi{bands_str}_aug"
    suffix += "_excl" if n_classes == 5 else ""
    suffix += f"_{split}" if split != "v1" else ""
    suffix += "_rot90"

    split_suffix = f"_{split}" if split != "v1" else ""
    test_halos = np.load(f"{thesis_root}/test_halos{split_suffix}.npy")

    with open(f"{thesis_root}/Scalers/scaler{suffix}.pkl", "rb") as f:
        band_scalers = pickle.load(f)["band_scalers"]

    test_data_list = []
    test_files_ref = None
    test_labels_str_ref = None

    for ch, band in enumerate(bands):
        filter_app = "" if band == "VIS" else f"_{band}"
        data_path = f"{images_root}/input_images{filter_app}_{sb}_f/"

        test_data, test_labels_str, test_files = get_data(
            data_path, test_halos, dm_models, snaps, apply_noise=False, exclude=excluded
        )

        if ch == 0:
            test_files_ref = test_files
            test_labels_str_ref = test_labels_str
        else:
            ref_basenames = [os.path.basename(f) for f in test_files_ref]
            cur_basenames = [os.path.basename(f) for f in test_files]
            assert ref_basenames == cur_basenames, f"File mismatch in {band} test set"

        info = band_scalers[band]
        scaled = info["scaler"].transform(test_data)
        scaled = (scaled - info["post_mean"]) / info["post_std"]
        test_data_list.append(scaled)

    test_multi = np.stack(test_data_list, axis=1).astype(np.float32)
    test_labels_int = label_str2int(test_labels_str_ref, dm_models)

    test_set = GalaxyDataset(test_multi, test_labels_int, augmentation=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Built multi-band test loader:")
    print(f"  Bands: {bands}")
    print(f"  Test images: {len(test_multi)}")
    print(f"  Shape per sample: {test_multi.shape[1:]}")
    print(f"  Class distribution: {np.bincount(test_labels_int)}")

    return test_loader, test_files_ref, suffix


def load_and_preprocess(fits_path, scaler, post_mean, post_std):
    """Load a single FITS image, apply a fitted scaler + standardization.

    Returns a tensor of shape (1, 1, H, W), ready for a single-image
    forward pass.
    """
    import torch
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        img = hdul[0].data.copy().astype(np.float32)

    img_scaled = scaler.transform(img)
    img_normalized = (img_scaled - post_mean) / post_std
    return torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def get_same_halo_images(
    halo: str, snap: int, los: int, dm_models: Sequence[str], sb_limit: int,
    scaler, post_mean: float, post_std: float,
    images_root: Optional[str] = None,
) -> dict:
    """Load the same halo/snapshot/LOS under each DM model, for comparison.

    Args:
        images_root: Parent of the `input_images_{sb_limit}_f` folder.
            Defaults to the original UZH cluster path.

    Returns:
        Dict mapping each DM model name to its preprocessed image tensor
        (or None if that file is missing).
    """
    images_root = images_root or UZH_ROOT
    base = f"{images_root}/input_images_{sb_limit}_f"
    halo_nr = halo.split("_")[1]

    images = {}
    for dm in dm_models:
        path = f"{base}/{halo}/{snap:03d}/SB_{halo_nr}_{dm}_{snap:03d}_LOS{los:02d}_N{sb_limit}.fits.gz"
        try:
            images[dm] = load_and_preprocess(path, scaler, post_mean, post_std)
        except FileNotFoundError:
            print(f"Missing: {path}")
            images[dm] = None
    return images