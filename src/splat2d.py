r"""2D Gaussian splatting renderer in pure PyTorch (Stage 1).

Coordinates
-----------
Continuous pixel coordinates ``(x, y)``: x to the right, y down, origin at the top-left
*corner* of the image. Pixel ``(row i, col j)`` covers ``[j, j+1) x [i, i+1)`` and is
sampled at its center ``p = (j + 0.5, i + 0.5)``. Flat pixel index is ``i * W + j``.

Gaussian k
----------
    mean        mu_k  in R^2               (pixels)
    rotation    theta_k                     (radians, counter-clockwise in the x-y frame*)
    scales      sigma_k = (s1, s2) > 0      (pixels, std. dev. along the rotated axes)
    color       c_k in [0, 1]^3
    opacity     o_k in (0, 1)

    Sigma_k = R(theta) diag(s1^2, s2^2) R(theta)^T,   R = [[cos, -sin], [sin, cos]]
    q_k(p)  = (p - mu)^T Sigma_k^{-1} (p - mu) = (u / s1)^2 + (v / s2)^2,
              where (u, v) = R^T (p - mu)  are the coordinates in the Gaussian's own frame
    G_k(p)  = exp(-q_k(p) / 2)              (unnormalized: peak value 1 at the mean)

(*with y pointing down, positive theta appears clockwise on screen.)
Computing q in the Gaussian frame avoids forming or inverting Sigma explicitly.

Support: G is truncated to the 3-sigma ellipse q < 9 (G > e^{-4.5} ~ 0.011 inside).

Blending
--------
additive:   C(p) = sum_k  o_k G_k(p) c_k                                   (order-free, unbounded)
alpha:      alpha_k(p) = min(0.99, o_k G_k(p))
            C(p) = sum_k c_k alpha_k T_k  +  T_final * bg,   T_k = prod_{j<k} (1 - alpha_j)
            with Gaussians visited front-to-back in a fixed depth order (here: index order,
            Gaussian 0 is in front). Early termination as in 3DGS: Gaussian k is skipped
            (and so is everything behind it) once T_k (1 - alpha_k) < t_min.

Sparse evaluation
-----------------
Instead of an (N, H*W) dense tensor we enumerate the (Gaussian, pixel) *pairs* inside
each Gaussian's 3-sigma ellipse. For alpha blending the pairs are sorted by
(pixel, depth); the transmittance is then a *segmented exclusive cumulative sum* in
log space:
    log T_k = sum_{j<k, same pixel} log(1 - alpha_j),
computed as a global cumsum minus the cumsum at the segment start. The global cumsum
runs over millions of pairs and reaches ~ -1e6, so it is done in float64 to avoid
catastrophic cancellation (in float32 the subtraction would lose ~0.1 in log T).
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.rasterize import ALPHA_MAX, Q_CUTOFF, rasterize


def conics_from_rotation(theta: Tensor, scales: Tensor) -> Tensor:
    """Upper triangle (A, B, C) of Sigma^{-1} = R diag(1/s1^2, 1/s2^2) R^T. (N, 3).

    With c = cos(theta), s = sin(theta), R = [[c, -s], [s, c]]:
        A = c^2/s1^2 + s^2/s2^2,  B = c s (1/s1^2 - 1/s2^2),  C = s^2/s1^2 + c^2/s2^2.
    """
    c, s = torch.cos(theta), torch.sin(theta)
    i1, i2 = scales[:, 0] ** -2, scales[:, 1] ** -2
    return torch.stack([c * c * i1 + s * s * i2, c * s * (i1 - i2), s * s * i1 + c * c * i2], -1)


def _mahalanobis_sq(px: Tensor, py: Tensor, mx: Tensor, my: Tensor, theta: Tensor,
                    s1: Tensor, s2: Tensor) -> Tensor:
    """q = (u/s1)^2 + (v/s2)^2 with (u, v) = R(theta)^T (p - mu), p the pixel center.

    Independent of the conic form; used by the dense reference.
    """
    dx = px.to(mx.dtype) + 0.5 - mx
    dy = py.to(my.dtype) + 0.5 - my
    c, s = torch.cos(theta), torch.sin(theta)
    u = c * dx + s * dy          # R^T d, first row:  ( cos, sin)
    v = -s * dx + c * dy         #        second row: (-sin, cos)
    return (u / s1) ** 2 + (v / s2) ** 2


def render(means: Tensor, theta: Tensor, scales: Tensor, colors: Tensor, opacities: Tensor,
           H: int, W: int, mode: str = "alpha", bg: Tensor | None = None,
           t_min: float = 1e-4, max_radius: float = 64.0) -> tuple[Tensor, dict]:
    """Render N 2D Gaussians to an (H, W, 3) image via :func:`src.rasterize.rasterize`.

    Args:
        means (N, 2), theta (N,), scales (N, 2) > 0, colors (N, 3), opacities (N,) in (0, 1):
            *activated* parameters (see module docstring for conventions).
        mode: "additive" or "alpha" (front-to-back in index order, Gaussian 0 in front).
        bg: (3,) background color for alpha mode (default black).
        t_min: early-termination threshold on transmittance (alpha mode); 0 disables it.
        max_radius: cap (pixels) on the 3-sigma extent; larger Gaussians are truncated.
    Returns:
        image (H, W, 3) in the parameters' dtype, and stats {"pairs": P}.
    """
    radii = (3.0 * scales.detach().amax(-1)).clamp(max=max_radius)
    img, _, info = rasterize(means, conics_from_rotation(theta, scales), radii, colors,
                             opacities, H, W, depths=None, bg=bg, mode=mode, t_min=t_min)
    return img, info


def render_dense(means: Tensor, theta: Tensor, scales: Tensor, colors: Tensor,
                 opacities: Tensor, H: int, W: int, mode: str = "alpha",
                 bg: Tensor | None = None, t_min: float = 1e-4) -> Tensor:
    """O(N * H * W) reference implementation of :func:`render` for tests (tiny scenes only).

    Evaluates every Gaussian at every pixel and composites with an explicit cumprod.
    No radius clamp, so it matches ``render`` whenever no Gaussian hits ``max_radius``.
    """
    ys, xs = torch.meshgrid(torch.arange(H, device=means.device),
                            torch.arange(W, device=means.device), indexing="ij")
    px, py = xs.reshape(-1), ys.reshape(-1)                               # (HW,)
    N, P = means.shape[0], px.numel()
    gid = torch.arange(N, device=means.device).repeat_interleave(P)       # (N*HW,)
    q = _mahalanobis_sq(px.repeat(N), py.repeat(N), means[gid, 0], means[gid, 1], theta[gid],
                        scales[gid, 0], scales[gid, 1]).view(N, P)
    G = torch.where(q < Q_CUTOFF, torch.exp(-0.5 * q), 0.0)
    w = opacities[:, None] * G                                             # (N, HW)

    if mode == "additive":
        img = torch.einsum("np,nc->pc", w, colors)
        return img.view(H, W, 3)

    alpha = w.clamp(max=ALPHA_MAX)
    one_minus = 1.0 - alpha
    T = torch.cumprod(torch.cat([torch.ones_like(one_minus[:1]), one_minus[:-1]]), dim=0)
    keep = (T * one_minus) >= t_min if t_min > 0 else torch.ones_like(T, dtype=torch.bool)
    contrib = torch.where(keep, T * alpha, 0.0)
    img = torch.einsum("np,nc->pc", contrib, colors)
    T_final = torch.prod(torch.where(keep, one_minus, 1.0), dim=0)
    bg = means.new_zeros(3) if bg is None else bg.to(means)
    return (img + T_final[:, None] * bg).view(H, W, 3)
