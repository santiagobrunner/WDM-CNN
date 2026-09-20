import torch
import pytest

from wdm_cnn.models.losses import gaussian_nll_with_components, moment_matching_loss


def test_gaussian_nll_components_sum_to_total():
    mu = torch.tensor([1.0, 2.0, 3.0])
    log_std = torch.tensor([0.1, -0.2, 0.3])
    target = torch.tensor([1.5, 1.8, 3.2])

    total, mean_term, var_term = gaussian_nll_with_components(mu, log_std, target)
    torch.testing.assert_close(total, mean_term + var_term)


def test_gaussian_nll_matches_manual_formula():
    """Cross-check against the standard per-element Gaussian NLL formula
    (full=False, i.e. omitting the constant 0.5*log(2*pi) term)."""
    mu = torch.tensor([0.5, -1.0, 2.0])
    log_std = torch.tensor([0.0, 0.5, -0.3])
    target = torch.tensor([0.0, 0.0, 0.0])

    total, _, _ = gaussian_nll_with_components(mu, log_std, target)

    var = torch.exp(log_std) ** 2
    expected = (0.5 * (torch.log(var) + (mu - target) ** 2 / var)).mean()
    torch.testing.assert_close(total, expected)


def test_gaussian_nll_does_not_explode_for_very_negative_log_std():
    """Regression test for the eps-clamping fix: without it, a very
    negative log_std drives var toward 0, causing division blow-up."""
    mu = torch.tensor([1.0])
    log_std = torch.tensor([-50.0])  # var would be ~0 without clamping
    target = torch.tensor([0.0])

    total, mean_term, var_term = gaussian_nll_with_components(mu, log_std, target)
    assert torch.isfinite(total)
    assert torch.isfinite(mean_term)
    assert torch.isfinite(var_term)


def test_moment_matching_loss_total_equals_weighted_sum():
    mu = torch.tensor([1.0, 2.0, 3.0])
    log_std = torch.tensor([0.1, -0.2, 0.3])
    target = torch.tensor([1.5, 1.8, 3.2])

    total, loss_mu, loss_var = moment_matching_loss(mu, log_std, target, alpha=2.0)
    torch.testing.assert_close(total, loss_mu + 2.0 * loss_var)


def test_moment_matching_loss_zero_residual_and_matched_variance():
    """If mu exactly matches target and var is driven very small, loss_mu
    (log of the near-zero squared residual) should approach log(eps)."""
    import math

    target = torch.tensor([1.0, 2.0, 3.0])
    mu = target.clone()
    log_std = torch.full_like(target, -30.0)

    total, loss_mu, loss_var = moment_matching_loss(mu, log_std, target, eps=1e-8)
    assert loss_mu.item() == pytest.approx(math.log(1e-8), abs=1e-4)