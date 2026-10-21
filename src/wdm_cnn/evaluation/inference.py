"""Running trained classifiers on a halo split and collecting their outputs.

Replaces the setup block that was copy-pasted into run_n_scan,
run_n_scan_multiband and run_n_scan_binary (build the suffix, load the
weights / scaler / post-stretch stats, load the images, run the model).

The result keeps LOGITS, not just probabilities: temperature scaling
(calibration.py) needs them. Probabilities are one call away:

    pred = load_predictions(thesis_root=..., images_root=..., data_type="test", ...)
    pred.probs()                    # plain softmax
    pred.probs(temperature=T)       # calibrated, with T fitted on a val split

torch is imported lazily, so importing this module (and using
`Predictions`) does not require it.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from wdm_cnn.evaluation.aggregation import halos_and_snaps
from wdm_cnn.evaluation.calibration import softmax


@dataclass
class Predictions:
    """Model outputs for one halo split, aligned image by image.

    Attributes:
        logits: Raw model outputs, shape (N, K).
        labels: True integer class labels, shape (N,).
        files: Image filenames, shape (N,). Row i of every array refers
            to the same image.
        class_names: The K class names, in class-index order.
        suffix: Model suffix (identifies the trained model).
    """

    logits: NDArray[np.float64]
    labels: NDArray[np.int_]
    files: NDArray
    class_names: list[str]
    suffix: str

    def __post_init__(self) -> None:
        self.logits = np.asarray(self.logits, dtype=float)
        self.labels = np.asarray(self.labels, dtype=int)
        self.files = np.asarray(self.files)
        if self.logits.ndim != 2:
            raise ValueError(f"logits must be (N, K), got shape {self.logits.shape}")
        n = len(self.logits)
        if len(self.labels) != n or len(self.files) != n:
            raise ValueError(
                f"logits, labels and files must have equal length, got "
                f"{n}, {len(self.labels)}, {len(self.files)}"
            )
        if self.logits.shape[1] != len(self.class_names):
            raise ValueError(
                f"logits have {self.logits.shape[1]} columns but {len(self.class_names)} class names"
            )

    def probs(self, temperature: float = 1.0) -> NDArray[np.float64]:
        """Class probabilities softmax(logits / temperature), shape (N, K)."""
        return softmax(self.logits, temperature)

    @property
    def halos(self) -> NDArray:
        """Halo name (e.g. 'h15') of each image, parsed from its filename."""
        return halos_and_snaps(self.files)[0]

    @property
    def snaps(self) -> NDArray:
        """Snapshot number of each image, parsed from its filename."""
        return halos_and_snaps(self.files)[1]

    def metadata(self, **extra) -> dict:
        """Dict for `scan_n(..., metadata=...)`: the suffix plus any extra columns."""
        return {"model_suffix": self.suffix, **extra}


def predict_logits(model, loader, device="cpu") -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Run `model` over `loader`; return (logits, labels) as numpy arrays.

    Rows come out in loader order, so pass a loader that does not shuffle
    (and does not augment) if the rows must line up with a list of files.
    """
    import torch

    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu().numpy())
            labels.append(y.cpu().numpy())
    if not logits:
        raise ValueError("loader produced no batches")
    return np.concatenate(logits).astype(np.float64), np.concatenate(labels).astype(int)


def load_model(
    model_name: str,
    n_classes: int,
    in_channels: int,
    weights_path: str,
    device="cpu",
    model_kwargs: Optional[dict] = None,
):
    """Build a model, load its saved weights, and put it in eval mode.

    Args:
        model_name: 'SimpleCNN', 'CNN5_CLS' or 'ConfigCNN'.
        in_channels: Number of input channels (1 single band; 3 for the
            'AllSc' scaler input). CNN5_CLS only supports 1.
        weights_path: A state_dict saved by the training script.
        model_kwargs: Extra constructor arguments, for ConfigCNN's
            architecture hyperparameters (n_blocks, base_channels, ...).
    """
    import torch

    from wdm_cnn.models.models import CNN5_CLS, ConfigCNN, SimpleCNN

    kwargs = dict(model_kwargs or {})
    if model_name == "SimpleCNN":
        net = SimpleCNN(n_classes=n_classes, in_channels=in_channels, **kwargs)
    elif model_name == "CNN5_CLS":
        if in_channels != 1:
            raise ValueError(f"CNN5_CLS supports 1 input channel only, got {in_channels}")
        net = CNN5_CLS(n_classes=n_classes, **kwargs)
    elif model_name == "ConfigCNN":
        net = ConfigCNN(n_classes=n_classes, in_channels=in_channels, **kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. Expected 'SimpleCNN', 'CNN5_CLS' or 'ConfigCNN'."
        )

    net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return net.to(device).eval()


def load_predictions(
    *,
    thesis_root: str,
    images_root: str,
    data_type: str = "test",
    SB_limit: int = 35,
    n_classes: int = 3,
    model: str = "SimpleCNN",
    band: str = "VIS",
    split: str = "v1",
    snapshots: str = "all",
    scaler_str: str = "shl",
    augment: bool = True,
    apply_exclusions: bool = False,
    rotation: str = "rot90",
    online: bool = False,
    masked: bool = False,
    res_factor: int = 1,
    suffix: Optional[str] = None,
    model_kwargs: Optional[dict] = None,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Predictions:
    """Load a trained model and its data, run it, and return aligned outputs.

    The paths have no defaults on purpose: the training script and the
    loaders have used different cluster roots, and a silent default would
    look for the weights, scaler or images in the wrong place.

    The other defaults match the thesis 3-class runs (train.py with
    --augment, --snapshots all, --rotation rot90, no --exclude). They
    must match how the model was trained, because together they form the
    suffix that locates its weights, scaler and post-stretch stats. For
    5-class models pass apply_exclusions=True.

    Args:
        data_type: Which halo split file to evaluate: 'train', 'val' or
            'test'. Use 'val' to fit a temperature or choose a binary
            threshold, 'test' to report results.
        split: Halo-split version ('v1', 'v2', ...), one per random split.
        suffix: Use this suffix as is instead of building it (for models
            without a suffix convention yet, e.g. ConfigCNN).
        model_kwargs: Architecture hyperparameters, for ConfigCNN.
        device: 'cuda' or 'cpu'; default: cuda if available.

    Returns:
        A `Predictions` with logits, labels, files, class names, suffix.

    Raises:
        RuntimeError: If the returned labels do not match the loaded ones
            (rows misaligned with the files).
    """
    import torch
    from torch.utils.data import DataLoader

    from wdm_cnn.data.loaders import dm_models_for, load_split_arrays
    from wdm_cnn.models.models import GalaxyDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_norm, targets, files, suffix = load_split_arrays(
        data_type=data_type, SB_limit=SB_limit, snapshots=snapshots, n_classes=n_classes,
        scaler_str=scaler_str, augment=augment, model=model, thesis_root=thesis_root,
        images_root=images_root, apply_exclusions=apply_exclusions, split=split,
        online=online, rotation=rotation, band=band, masked=masked,
        res_factor=res_factor, suffix=suffix,
    )

    in_channels = 1 if data_norm.ndim == 3 else data_norm.shape[1]
    net = load_model(
        model, n_classes, in_channels, f"{thesis_root}/Model_Params/best_model{suffix}.pth",
        device=device, model_kwargs=model_kwargs,
    )

    # No shuffling and no augmentation: row i of the output is file i.
    loader = DataLoader(
        GalaxyDataset(data_norm, targets, augmentation=False), batch_size=batch_size, shuffle=False
    )
    logits, labels = predict_logits(net, loader, device)

    if not np.array_equal(labels, targets):
        raise RuntimeError("model outputs are not aligned with the loaded files (labels differ)")

    return Predictions(
        logits=logits, labels=labels, files=files,
        class_names=dm_models_for(n_classes), suffix=suffix,
    )