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

import math

import torch
from torch import Tensor

ALPHA_MAX = 0.99   # clamp so log(1 - alpha) stays finite and every layer lets some light through
Q_CUTOFF = 9.0     # 3-sigma ellipse


def screen_radius(scales: Tensor) -> Tensor:
    """Integer half-width (pixels) of the box that contains the 3-sigma ellipse. (N,) int64.

    Pixel indices ``floor(mu) +- r`` with ``r = ceil(3 max(s1, s2))`` contain every pixel
    center within 3*max(s) of mu (proof: |j + 0.5 - mu| <= 3s  =>  |j - floor(mu)| <= ceil(3s)).
    """
    return torch.ceil(3.0 * scales.detach().amax(dim=-1)).long()


def enumerate_pairs(means: Tensor, theta: Tensor, scales: Tensor, H: int, W: int,
                    max_radius: int = 64, max_pairs: int = 60_000_000,
                    sort_by_pixel: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """Integer (Gaussian, pixel) pairs with q < 9. No autograd; returns (gid, px, py), each (P,).

    means (N, 2) px, theta (N,), scales (N, 2) px. Box radius is clamped to ``max_radius``
    (a very large Gaussian is truncated - its gradients then ignore the cut-off part).
    With ``sort_by_pixel`` the pairs are ordered by (pixel, Gaussian index), i.e. per pixel
    front-to-back. Sorting the integers here, before any differentiable computation, keeps
    the permutation out of the autograd graph.
    """
    N = means.shape[0]
    dev = means.device
    means, theta, scales = means.detach(), theta.detach(), scales.detach()
    with torch.no_grad():
        r = screen_radius(scales).clamp(max=max_radius)
        side = 2 * r + 1
        counts = side * side
        total = int(counts.sum())
        if total > max_pairs:
            raise RuntimeError(f"{total:,} candidate pairs exceeds max_pairs={max_pairs:,}; "
                               "Gaussians are too large (lower max_radius or regularize scales)")
        gid = torch.repeat_interleave(torch.arange(N, device=dev), counts, output_size=total)
        start = torch.cumsum(counts, 0) - counts
        k = torch.arange(total, device=dev) - start[gid]
        s, rr = side[gid], r[gid]
        center = torch.floor(means).long()
        px = center[gid, 0] + (k % s) - rr
        py = center[gid, 1] + (k // s) - rr
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        gid, px, py = gid[ok], px[ok], py[ok]
        q = _mahalanobis_sq(px, py, means[gid, 0], means[gid, 1], theta[gid],
                            scales[gid, 0], scales[gid, 1])
        ok = q < Q_CUTOFF
        gid, px, py = gid[ok], px[ok], py[ok]
        if sort_by_pixel:
            order = torch.argsort((py * W + px) * N + gid)
            gid, px, py = gid[order], px[order], py[order]
    return gid, px, py


def _mahalanobis_sq(px: Tensor, py: Tensor, mx: Tensor, my: Tensor, theta: Tensor,
                    s1: Tensor, s2: Tensor) -> Tensor:
    """q = (u/s1)^2 + (v/s2)^2 with (u, v) = R(theta)^T (p - mu), p the pixel center.

    All arguments are per pair, shape (P,).
    """
    dx = px.to(mx.dtype) + 0.5 - mx
    dy = py.to(my.dtype) + 0.5 - my
    c, s = torch.cos(theta), torch.sin(theta)
    u = c * dx + s * dy          # R^T d, first row:  ( cos, sin)
    v = -s * dx + c * dy         #        second row: (-sin, cos)
    return (u / s1) ** 2 + (v / s2) ** 2


def _gather(gid: Tensor, means, theta, scales, colors, opacities) -> Tensor:
    """Per-pair copy of all Gaussian parameters, (P, 9): mx, my, theta, s1, s2, r, g, b, o.

    One ``index_select`` on a packed (N, 9) tensor: its backward is a single atomic
    ``index_add``, much cheaper than the sort-based backward of several ``x[gid]`` gathers.
    """
    packed = torch.cat([means, theta[:, None], scales, colors, opacities[:, None]], dim=1)
    return packed.index_select(0, gid)


def render(means: Tensor, theta: Tensor, scales: Tensor, colors: Tensor, opacities: Tensor,
           H: int, W: int, mode: str = "alpha", bg: Tensor | None = None,
           t_min: float = 1e-4, max_radius: int = 64) -> tuple[Tensor, dict]:
    """Render N 2D Gaussians to an (H, W, 3) image.

    Args:
        means (N, 2), theta (N,), scales (N, 2) > 0, colors (N, 3), opacities (N,) in (0, 1):
            *activated* parameters (see module docstring for conventions).
        mode: "additive" or "alpha" (front-to-back in index order, Gaussian 0 in front).
        bg: (3,) background color for alpha mode (default black).
        t_min: early-termination threshold on transmittance (alpha mode); 0 disables it.
    Returns:
        image (H, W, 3) in the parameters' dtype, and stats {"pairs": P}.
    """
    if mode not in ("additive", "alpha"):
        raise ValueError(f"unknown mode {mode!r}")
    gid, px, py = enumerate_pairs(means, theta, scales, H, W, max_radius=max_radius,
                                  sort_by_pixel=(mode == "alpha"))
    pix = py * W + px
    g = _gather(gid, means, theta, scales, colors, opacities)
    q = _mahalanobis_sq(px, py, g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4])
    w = g[:, 8] * torch.exp(-0.5 * q)
    rgb = g[:, 5:8]
    out = means.new_zeros(H * W, 3)

    if mode == "additive":
        out = out.index_add(0, pix, w[:, None] * rgb)
        return out.view(H, W, 3), {"pairs": pix.numel()}

    # Pairs are already sorted by (pixel, depth); depth order == Gaussian index.
    alpha = w.clamp(max=ALPHA_MAX)

    # Segmented exclusive cumsum of log(1 - alpha) per pixel, in float64.
    L = torch.log1p(-alpha.double())
    incl = torch.cumsum(L, 0)
    excl = incl - L
    is_start = torch.ones_like(pix, dtype=torch.bool)
    is_start[1:] = pix[1:] != pix[:-1]
    seg = torch.cumsum(is_start.long(), 0) - 1               # segment id for each pair
    start_idx = torch.nonzero(is_start).squeeze(1)
    log_T = excl - excl.index_select(0, start_idx).index_select(0, seg)   # exclusive, per pixel

    # Early termination: keep k iff T_k (1 - alpha_k) >= t_min. T is non-increasing along a
    # segment, so this keeps a prefix of each pixel's list - exactly the sequential rule.
    if t_min > 0:
        keep = (log_T + L).detach() >= math.log(t_min)
    else:
        keep = torch.ones_like(L, dtype=torch.bool)

    T = torch.exp(log_T).to(alpha.dtype)
    contrib = torch.where(keep, T * alpha, 0.0)
    out = out.index_add(0, pix, contrib[:, None] * rgb)

    # Final transmittance per pixel: exp(sum of kept log(1 - alpha)); 1 where nothing landed.
    log_T_final = torch.zeros(H * W, dtype=L.dtype, device=L.device)
    log_T_final = log_T_final.index_add(0, pix, torch.where(keep, L, 0.0))
    T_final = torch.exp(log_T_final).to(alpha.dtype)
    bg = means.new_zeros(3) if bg is None else bg.to(means)
    out = out + T_final[:, None] * bg
    return out.view(H, W, 3), {"pairs": pix.numel()}


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
