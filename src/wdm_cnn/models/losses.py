"""Loss functions for the RegrCNN regression extension (predicting
1/m_WDM directly, rather than classifying discrete DM models). This
extension goes beyond the thesis's classification scope.
"""

import torch


def moment_matching_loss(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Moment-matching loss: trains mu on squared residual, and log_std
    so that the predicted variance matches the observed squared residual
    (second moment), rather than via a standard likelihood.

    Args:
        mu: Predicted mean, shape (B,).
        log_std: Predicted log standard deviation, shape (B,).
        target: True target values, shape (B,).
        alpha: Weight on the variance-matching term relative to the
            mean term.
        eps: Added inside each log() to avoid log(0).

    Returns:
        (total_loss, loss_mu, loss_var).
    """
    var = torch.exp(log_std) ** 2
    sq_resid = (mu - target) ** 2

    loss_mu = torch.log(sq_resid.mean() + eps)
    loss_var = torch.log(((sq_resid - var) ** 2).mean() + eps)
    total = loss_mu + alpha * loss_var
    return total, loss_mu, loss_var


def gaussian_nll_with_components(
    mu: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaussian negative log-likelihood, with the mean and variance
    terms also returned separately.

    Matches torch.nn.GaussianNLLLoss with full=False (the constant
    0.5*log(2*pi) term is omitted, as it doesn't affect gradients).

    Args:
        mu: Predicted mean, shape (B,).
        log_std: Predicted log standard deviation, shape (B,).
        target: True target values, shape (B,).
        eps: Minimum allowed variance, to avoid division by ~0 / log(0)
            if the model predicts a very negative log_std. Matches the
            role of PyTorch's own `eps` parameter for this loss (not
            present in the original version of this function).

    Returns:
        (total_loss, mean_term, variance_term), where
        total_loss == mean_term + variance_term.
    """
    var = torch.clamp(torch.exp(log_std) ** 2, min=eps)
    log_var = torch.log(var)

    loss_mu = ((mu - target) ** 2 / var).mean()
    loss_var = log_var.mean()
    total = 0.5 * (loss_mu + loss_var)
    return total, 0.5 * loss_mu, 0.5 * loss_var