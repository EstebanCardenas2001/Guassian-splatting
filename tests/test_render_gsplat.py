"""Stage 5: gsplat wrapper agrees with the pure-PyTorch renderer (images and gradients).

render_torch is run with gsplat's constants (alpha clamp 0.999, support q < 3.33^2) and
opacities <= 0.95, so the only remaining differences are float32 arithmetic order and the
CUDA fast exp (__expf).
"""
import math

import pytest
import torch

from src.camera import look_at, intrinsics_matrix
from src.render_gsplat import GSPLAT_ALPHA_MAX, GSPLAT_Q_CUTOFF
from src.render_gsplat import render as render_gs
from src.render_torch import render as render_pt

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat needs CUDA")
DEV = "cuda"


def scene(N=300, seed=0, extent=1.0, scale=(0.02, 0.2), opacity=(0.1, 0.95)):
    g = torch.Generator().manual_seed(seed)
    u = lambda *s: torch.rand(*s, generator=g)
    means = (2 * u(N, 3) - 1) * extent
    quats = torch.randn(N, 4, generator=g)
    scales = scale[0] + (scale[1] - scale[0]) * u(N, 3)
    opac = opacity[0] + (opacity[1] - opacity[0]) * u(N)
    colors = u(N, 3)
    return [t.to(DEV) for t in (means, quats, scales, opac, colors)]


def camera(W=96, H=72, f=90.0, eye=(0.5, 1.0, -4.0)):
    V = look_at(torch.tensor(eye), torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]))
    return V.to(DEV), intrinsics_matrix(f, f, W / 2, H / 2).to(DEV), W, H


def both(params, cam, bg, antialiased=False):
    V, K, W, H = cam
    a = render_pt(*params, V, K, W, H, bg=bg, antialiased=antialiased,
                  alpha_max=GSPLAT_ALPHA_MAX, q_cutoff=GSPLAT_Q_CUTOFF)
    b = render_gs(*params, V, K, W, H, bg=bg, antialiased=antialiased)
    return a, b


@pytest.mark.parametrize("antialiased", [False, True])
@pytest.mark.parametrize("eye", [(0.5, 1.0, -4.0), (3.0, -1.0, 2.0)])
def test_images_agree(antialiased, eye):
    params = scene()
    bg = torch.tensor([1.0, 1.0, 1.0], device=DEV)
    (ia, aa, _), (ib, ab, _) = both(params, camera(eye=eye), bg, antialiased)
    assert (ia - ib).abs().max() < 1e-5, (ia - ib).abs().max()   # measured ~1e-6 (float32)
    assert (ia - ib).abs().mean() < 1e-6
    assert (aa - ab).abs().max() < 1e-5


def test_culling_agrees():
    params = scene(N=50)
    V, K, W, H = camera()
    (_, _, a), (_, _, b) = both(params, (V, K, W, H), None)
    # radii differ by definition (gsplat rounds per-axis extents up), visibility must not
    assert torch.equal(a["valid"] & (a["radii"] > 1.5), b["valid"] & (a["radii"] > 1.5))


def test_packed_equals_unpacked():
    params = scene()
    V, K, W, H = camera()
    a = render_gs(*params, V, K, W, H, packed=True)[0]
    b = render_gs(*params, V, K, W, H, packed=False)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_gradients_agree():
    """Same random linear functional of the image, gradients w.r.t. all five parameter groups."""
    base = scene(N=120, seed=3)
    bg = torch.tensor([0.2, 0.5, 0.8], device=DEV)
    V, K, W, H = camera()
    w = torch.randn(H, W, 3, generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
    grads = []
    for fn, kw in ((render_pt, dict(alpha_max=GSPLAT_ALPHA_MAX, q_cutoff=GSPLAT_Q_CUTOFF)), (render_gs, {})):
        p = [t.clone().requires_grad_() for t in base]
        img = fn(*p, V, K, W, H, bg=bg, **kw)[0]
        (img * w).sum().backward()
        grads.append([t.grad for t in p])
    for name, ga, gb in zip(("means", "quats", "scales", "opacities", "colors"), *grads):
        cos = torch.nn.functional.cosine_similarity(ga.flatten(), gb.flatten(), dim=0)
        rel = (ga - gb).norm() / ga.norm()
        assert cos > 0.99999 and rel < 1e-3, (name, float(cos), float(rel))   # measured ~1e-5
