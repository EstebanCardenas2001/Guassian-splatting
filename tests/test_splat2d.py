"""Stage 1: sparse 2D splatting renderer vs dense reference, plus gradcheck."""
import math

import pytest
import torch

from src.splat2d import render, render_dense, screen_radius

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def random_scene(N, H, W, dtype=torch.float64, seed=0, opacity=(0.2, 0.9), scale=(0.8, 3.0)):
    g = torch.Generator().manual_seed(seed)
    u = lambda *s: torch.rand(*s, generator=g, dtype=dtype)
    means = u(N, 2) * torch.tensor([W, H], dtype=dtype)
    theta = u(N) * math.pi
    scales = scale[0] + u(N, 2) * (scale[1] - scale[0])
    colors = u(N, 3)
    opac = opacity[0] + u(N) * (opacity[1] - opacity[0])
    return [t.to(DEV) for t in (means, theta, scales, colors, opac)]


def test_single_gaussian_peak_and_axes():
    """An axis-aligned Gaussian at a pixel center: peak o*c, falls to exp(-1/2) at 1 sigma."""
    H = W = 21
    means = torch.tensor([[10.5, 10.5]], dtype=torch.float64, device=DEV)
    theta = torch.zeros(1, dtype=torch.float64, device=DEV)
    scales = torch.tensor([[2.0, 4.0]], dtype=torch.float64, device=DEV)
    colors = torch.tensor([[1.0, 0.5, 0.25]], dtype=torch.float64, device=DEV)
    opac = torch.tensor([0.8], dtype=torch.float64, device=DEV)
    img, _ = render(means, theta, scales, colors, opac, H, W, mode="additive")
    assert torch.allclose(img[10, 10], 0.8 * colors[0])
    # x is the column, scale 2 along x; y is the row, scale 4 along y
    assert torch.isclose(img[10, 12, 0], 0.8 * torch.exp(torch.tensor(-0.5, dtype=img.dtype)))
    assert torch.isclose(img[14, 10, 0], 0.8 * torch.exp(torch.tensor(-0.5, dtype=img.dtype)))
    # rotating by 90 degrees swaps the axes
    img90, _ = render(means, theta + math.pi / 2, scales, colors, opac, H, W, mode="additive")
    assert torch.allclose(img90, img.transpose(0, 1))


def test_radius_contains_ellipse():
    scales = torch.tensor([[0.3, 0.3], [2.5, 1.0], [10.0, 0.5]])
    assert screen_radius(scales).tolist() == [1, 8, 30]


@pytest.mark.parametrize("mode", ["additive", "alpha"])
@pytest.mark.parametrize("t_min", [0.0, 1e-4])
def test_sparse_matches_dense(mode, t_min):
    if mode == "additive" and t_min > 0:
        pytest.skip("early termination only applies to alpha blending")
    N, H, W = 60, 24, 32
    scene = random_scene(N, H, W)
    bg = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, device=DEV)
    a, _ = render(*scene, H, W, mode=mode, bg=bg, t_min=t_min)
    b = render_dense(*scene, H, W, mode=mode, bg=bg, t_min=t_min)
    assert torch.allclose(a, b, atol=1e-10), (a - b).abs().max()


def test_early_termination_is_active_and_matches():
    """Many nearly opaque, stacked Gaussians: termination must actually change the image."""
    N, H, W = 40, 8, 8
    scene = random_scene(N, H, W, opacity=(0.97, 0.99), scale=(6.0, 8.0), seed=3)
    off, _ = render(*scene, H, W, t_min=0.0)
    on, _ = render(*scene, H, W, t_min=1e-2)
    assert (off - on).abs().max() > 1e-4
    assert torch.allclose(on, render_dense(*scene, H, W, t_min=1e-2), atol=1e-10)


@pytest.mark.parametrize("mode", ["additive", "alpha"])
def test_gradcheck(mode):
    N, H, W = 6, 10, 12
    means, theta, scales, colors, opac = random_scene(N, H, W, seed=1)
    # avoid alpha hitting the 0.99 clamp, where the derivative is discontinuous
    opac = opac * 0.8
    params = [p.clone().requires_grad_() for p in (means, theta, scales, colors, opac)]
    bg = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float64, device=DEV)
    f = lambda *p: render(*p, H, W, mode=mode, bg=bg, t_min=0.0)[0]
    # index_select's backward is an atomic index_add: summation order varies run to run,
    # so repeated backward passes differ at the ~1e-16 level (allowed by nondet_tol).
    assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-6, nondet_tol=1e-12)
