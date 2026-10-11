"""SSIM against torchmetrics and basic properties."""
import pytest
import torch

from src.losses import photometric_loss, ssim

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_ssim_identity_is_one():
    x = torch.rand(32, 40, 3, device=DEV)
    assert torch.isclose(ssim(x, x), torch.tensor(1.0, device=DEV), atol=1e-6)
    assert photometric_loss(x, x).abs() < 1e-6


def test_ssim_matches_skimage_exactly_and_torchmetrics_loosely():
    """Reference: scikit-image (gaussian weights, sigma 1.5, population covariance), which
    agrees with this implementation to float64 precision. torchmetrics differs by ~3e-4
    (its reflect-pad-then-crop border handling), so it is only a loose cross-check."""
    sk = pytest.importorskip("skimage.metrics")
    tm = pytest.importorskip("torchmetrics.functional.image")
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 48, 64, 3, generator=g, dtype=torch.float64)
    y = (x + 0.1 * torch.randn(x.shape, generator=g, dtype=torch.float64)).clamp(0, 1)
    ours = ssim(y, x, padding="valid").item()
    ref = sk.structural_similarity(y[0].numpy(), x[0].numpy(), gaussian_weights=True, sigma=1.5,
                                   use_sample_covariance=False, data_range=1.0, channel_axis=-1)
    assert abs(ours - ref) < 1e-10
    loose = tm.structural_similarity_index_measure(y.permute(0, 3, 1, 2), x.permute(0, 3, 1, 2),
                                                   gaussian_kernel=True, sigma=1.5, kernel_size=11,
                                                   data_range=1.0).item()
    assert abs(ours - loose) < 2e-3


def test_ssim_decreases_with_noise():
    x = torch.rand(64, 64, 3, device=DEV)
    vals = [ssim((x + s * torch.randn_like(x)).clamp(0, 1), x).item() for s in (0.01, 0.05, 0.2)]
    assert vals[0] > vals[1] > vals[2]
