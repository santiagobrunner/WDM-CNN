"""Grad-CAM helpers for trained classifiers.

Requires the optional `grad-cam` package (PyPI name `grad-cam`, imported
as `pytorch_grad_cam`). It is imported lazily inside the functions that
need it, so importing this module does not require it.
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _last_conv_layer(model):
    """Return the last nn.Conv2d registered in `model`.

    Works for SimpleCNN (conv6), CNN5_CLS (conv5b) and ConfigCNN (last
    conv inside its `features` Sequential) alike, because modules are
    registered in definition order. This replaces the earlier
    hasattr(model, 'conv6') / hasattr(model, 'conv5b') chain, which
    could not handle ConfigCNN.
    """
    import torch.nn as nn

    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError(f"No Conv2d layer found in {type(model).__name__}")
    return last


def get_gradcam(model, image, cam=None, target_class=None, target_layers=None, aug_smooth=True):
    """Run Grad-CAM on a single image.

    Args:
        model: Trained CNN.
        image: Tensor of shape (1, C, H, W), already on the model's device.
        cam: A pytorch_grad_cam CAM object to reuse across images. If
            None, a GradCAMPlusPlus is created. NOTE: the thesis
            (Section 2.4.7, Eq. 56) describes plain Grad-CAM, so pass
            `cam=GradCAM(...)` explicitly if that is the variant you want.
        target_class: Class whose evidence is visualised. Defaults to the
            predicted class (which can differ from the true class).
        target_layers: Layers to hook. Defaults to the last Conv2d.
        aug_smooth: Average the CAM over augmented views of the input
            (the thesis Fig. 27 caption mentions averaging over
            augmented views).

    Returns:
        (heatmap, pred_class, confidence, probs): heatmap is (H, W);
        confidence is the probability of `target_class`; probs is the
        full softmax vector.
    """
    import torch
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if target_layers is None:
        target_layers = [_last_conv_layer(model)]

    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        if target_class is None:
            target_class = pred_class
        confidence = probs[0, target_class].item()

    if cam is None:
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    heatmap = cam(
        input_tensor=image,
        targets=[ClassifierOutputTarget(target_class)],
        aug_smooth=aug_smooth,
    )
    return heatmap[0], pred_class, confidence, probs[0].cpu().numpy()


def get_gradcams_by_label(model, cam, target_label, test_loader, dm_models, device, n=3):
    """Collect up to `n` Grad-CAMs from correctly classified images of one class.

    Args:
        target_label: DM-model name (e.g. 'cdm'); must be in `dm_models`.
        test_loader: Yields (x_batch, y_batch) with integer labels.

    Returns:
        (gradcams, images, preds, confidences), each a list of length
        <= n (shorter if the loader runs out of correct examples).
    """
    target_index = dm_models.index(target_label)
    gradcams, images, preds, confidences = [], [], [], []

    for x_batch, y_batch in test_loader:
        for i in range(len(y_batch)):
            if y_batch[i].item() != target_index:
                continue
            image = x_batch[i : i + 1].to(device)
            gradcam, pred, conf, _ = get_gradcam(model, image, cam=cam)
            if pred == target_index:
                gradcams.append(gradcam)
                images.append(image)
                preds.append(pred)
                confidences.append(conf)
                if len(gradcams) == n:
                    return gradcams, images, preds, confidences

    return gradcams, images, preds, confidences


def _to_numpy_image(image):
    """Accept a torch tensor or array; return a squeezed 2D numpy array."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    return np.squeeze(np.asarray(image))


def plot_gradcam(
    image,
    true_label: int,
    heatmap,
    pred: int,
    confidence: float,
    dm_models: Sequence[str],
    mask=None,
    save_path: Optional[str] = None,
    show: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot the input image, the Grad-CAM heatmap, and their overlay.

    Args:
        image: Preprocessed input as seen by the model (tensor or array).
        true_label, pred: Integer class indices into `dm_models`.
        confidence: Currently unused (the title no longer shows it); kept
            so existing call sites do not break.
        mask: Optional boolean array, same shape as `heatmap`. Its
            boundary is drawn on the original and overlay panels.
        save_path: If given, save the figure there (300 dpi, tight bbox).
        show: Call plt.show(); set False for batch runs (the figure is
            then closed after saving).
        vmin, vmax: Shared colour limits, to compare maps across images.

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    img_np = _to_numpy_image(image)

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    im1 = axes[1].imshow(heatmap, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Grad-CAM", fontsize=13)
    axes[1].axis("off")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Activation", fontsize=10)

    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(heatmap, cmap="inferno", alpha=0.75, vmin=vmin, vmax=vmax)
    if mask is not None:
        axes[0].contour(mask, levels=[0.5], colors="cyan", linewidths=1.1, alpha=0.5)
        axes[2].contour(mask, levels=[0.5], colors="cyan", linewidths=1.1, alpha=0.5)
    axes[2].set_title("Overlay", fontsize=13)
    axes[2].axis("off")

    status = "correct" if pred == true_label else "INCORRECT"
    fig.suptitle(
        f"pred: {dm_models[pred]}  /  true: {dm_models[true_label]}  [{status}]",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig