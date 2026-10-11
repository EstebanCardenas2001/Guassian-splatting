r"""Sparse, differentiable rasterization of 2D (screen-space) Gaussians in pure PyTorch.

Shared by the Stage 1 image fitter (``splat2d``) and the 3D renderer (``render_torch``):
both reduce to N screen-space Gaussians, each described by

    means2d  (N, 2)  center in pixel coordinates (x right, y down; pixel (i, j) has its
                     center at (j + 0.5, i + 0.5))
    conics   (N, 3)  (A, B, C): upper triangle of the inverse 2D covariance, so that
                     q(p) = A dx^2 + 2 B dx dy + C dy^2,   (dx, dy) = p - mean
    radii    (N,)    float extent in pixels (3 * sqrt(largest eigenvalue)); <= 0 means culled
    colors   (N, D)  any number of channels
    opacities (N,)
    depths   (N,)    sort key, smaller = in front (or None: index order, 0 in front)

and the rendered value at pixel p is (alpha mode)

    alpha_k = min(0.99, o_k exp(-q_k(p) / 2))      for q_k < 9 and alpha_k >= alpha_min
    C(p)    = sum_k c_k alpha_k T_k + T_final bg,   T_k = prod_{j<k} (1 - alpha_j)

with Gaussians visited front-to-back and 3DGS early termination
(stop before k once T_k (1 - alpha_k) < t_min).

Implementation: enumerate only (Gaussian, pixel) pairs inside each 3-sigma ellipse,
sort them by (pixel, depth) *as integers, before autograd sees them*, gather all
per-Gaussian quantities with one ``index_select`` (atomic backward), and compute
log T as a segmented exclusive cumsum in float64 (the global cumsum reaches ~ -1e6,
so float32 would lose ~0.1 in log T to cancellation).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

ALPHA_MAX = 0.99   # clamp so log(1 - alpha) stays finite and every layer lets some light through
Q_CUTOFF = 9.0     # 3-sigma ellipse


def conic_quadratic(dx: Tensor, dy: Tensor, conic: Tensor) -> Tensor:
    """q = A dx^2 + 2 B dx dy + C dy^2 for per-pair offsets and conic (P, 3)."""
    return conic[..., 0] * dx * dx + 2.0 * conic[..., 1] * dx * dy + conic[..., 2] * dy * dy


@torch.no_grad()
def enumerate_pairs(means2d: Tensor, conics: Tensor, radii: Tensor, H: int, W: int,
                    depths: Tensor | None, sort: bool, q_max: Tensor | None = None,
                    max_pairs: int = 80_000_000) -> tuple[Tensor, Tensor, Tensor]:
    """Integer (Gaussian, pixel) pairs with q < q_max (default 9, the 3-sigma ellipse;
    or a per-Gaussian (N,) tensor). Returns (gid, px, py), each (P,) int64.

    The candidate box of Gaussian k is the axis-aligned bounding box of its ellipse
    q < q_max, i.e. half-widths r_x = sqrt(q_max Sigma_xx), r_y = sqrt(q_max Sigma_yy) with
    Sigma = conic^{-1}: Sigma_xx = C / (AC - B^2), Sigma_yy = A / (AC - B^2). (For a thin,
    tilted splat this is far smaller than the square of half-width ``radii``, which is
    only used as a validity flag and upper bound.) Pixels whose *centers* are in the box:
        x in [ceil(mx - 0.5 - r_x), floor(mx - 0.5 + r_x)] ∩ [0, W-1]   (same for y).
    With ``sort``, pairs are ordered by (pixel, depth rank): per pixel, front to back.
    """
    N = means2d.shape[0]
    dev = means2d.device
    m, r, cn = means2d.detach(), radii.detach(), conics.detach()
    valid = (r > 0) & torch.isfinite(m).all(-1)
    r = torch.where(valid, r, 0.0)
    m = torch.where(valid[:, None], m, 0.0)
    qm = torch.full_like(r, Q_CUTOFF) if q_max is None else q_max.detach().clamp(min=0)
    det = (cn[:, 0] * cn[:, 2] - cn[:, 1] ** 2).clamp(min=1e-30)
    rx = torch.minimum(torch.sqrt(qm * cn[:, 2] / det), r)
    ry = torch.minimum(torch.sqrt(qm * cn[:, 0] / det), r)
    x0 = torch.ceil(m[:, 0] - 0.5 - rx).clamp(0, W).long()
    x1 = torch.floor(m[:, 0] - 0.5 + rx).clamp(-1, W - 1).long()
    y0 = torch.ceil(m[:, 1] - 0.5 - ry).clamp(0, H).long()
    y1 = torch.floor(m[:, 1] - 0.5 + ry).clamp(-1, H - 1).long()
    nx = (x1 - x0 + 1).clamp(min=0)
    ny = (y1 - y0 + 1).clamp(min=0)
    counts = torch.where(valid, nx * ny, 0)
    total = int(counts.sum())
    if total > max_pairs:
        raise RuntimeError(f"{total:,} candidate pairs exceeds max_pairs={max_pairs:,} "
                           "(Gaussians too large or too many)")
    gid = torch.repeat_interleave(torch.arange(N, device=dev), counts, output_size=total)
    k = torch.arange(total, device=dev) - (torch.cumsum(counts, 0) - counts)[gid]
    w = nx[gid]
    px = x0[gid] + k % w
    py = y0[gid] + k // w
    dx = px.to(m.dtype) + 0.5 - m[gid, 0]
    dy = py.to(m.dtype) + 0.5 - m[gid, 1]
    cut = Q_CUTOFF if q_max is None else q_max.detach()[gid]
    ok = conic_quadratic(dx, dy, conics.detach()[gid]) < cut
    gid, px, py = gid[ok], px[ok], py[ok]
    if sort:
        rank = (torch.arange(N, device=dev) if depths is None
                else torch.argsort(torch.argsort(depths.detach())))
        order = torch.argsort((py * W + px) * N + rank[gid])
        gid, px, py = gid[order], px[order], py[order]
    return gid, px, py


def rasterize(means2d: Tensor, conics: Tensor, radii: Tensor, colors: Tensor,
              opacities: Tensor, H: int, W: int, depths: Tensor | None = None,
              bg: Tensor | None = None, mode: str = "alpha", t_min: float = 1e-4,
              alpha_min: float = 0.0, q_max: Tensor | None = None,
              max_pairs: int = 80_000_000) -> tuple[Tensor, Tensor, dict]:
    """Render screen-space Gaussians (see module docstring). Differentiable w.r.t.
    means2d, conics, colors, opacities (and bg); radii, depths and q_max only select/sort.

    ``q_max`` (N,) optionally tightens the per-Gaussian support below q < 9 (e.g. to where
    alpha >= alpha_min, which leaves the image unchanged but evaluates fewer pairs).

    Returns:
        image (H, W, D), alpha (H, W) = 1 - T_final (alpha mode; additive: sum of weights),
        info {"pairs": number of (Gaussian, pixel) pairs evaluated}.
    """
    if mode not in ("additive", "alpha"):
        raise ValueError(f"unknown mode {mode!r}")
    D = colors.shape[-1]
    gid, px, py = enumerate_pairs(means2d, conics, radii, H, W, depths, sort=(mode == "alpha"),
                                  q_max=q_max, max_pairs=max_pairs)
    pix = py * W + px

    # One packed gather: (P, 2 + 3 + 1 + D) = mean, conic, opacity, color.
    packed = torch.cat([means2d, conics, opacities[:, None], colors], dim=1)
    g = packed.index_select(0, gid)
    dx = px.to(g.dtype) + 0.5 - g[:, 0]
    dy = py.to(g.dtype) + 0.5 - g[:, 1]
    q = conic_quadratic(dx, dy, g[:, 2:5])
    w = g[:, 5] * torch.exp(-0.5 * q)
    rgb = g[:, 6:6 + D]
    out = means2d.new_zeros(H * W, D)
    info = {"pairs": pix.numel()}

    if mode == "additive":
        out = out.index_add(0, pix, w[:, None] * rgb)
        acc = means2d.new_zeros(H * W).index_add(0, pix, w)
        return out.view(H, W, D), acc.view(H, W), info

    alpha = w.clamp(max=ALPHA_MAX)
    used = (alpha.detach() >= alpha_min) if alpha_min > 0 else torch.ones_like(alpha, dtype=torch.bool)

    # Segmented exclusive cumsum of log(1 - alpha) per pixel, in float64.
    L = torch.where(used, torch.log1p(-alpha.double()), 0.0)
    excl = torch.cumsum(L, 0) - L
    is_start = torch.ones_like(pix, dtype=torch.bool)
    is_start[1:] = pix[1:] != pix[:-1]
    seg = torch.cumsum(is_start.long(), 0) - 1
    start_idx = torch.nonzero(is_start).squeeze(1)
    log_T = excl - excl.index_select(0, start_idx).index_select(0, seg)

    # Early termination: keep k iff T_k (1 - alpha_k) >= t_min. T is non-increasing along a
    # pixel's list, so this keeps a prefix - exactly the sequential front-to-back rule.
    keep = used & ((log_T + L).detach() >= math.log(t_min)) if t_min > 0 else used

    T = torch.exp(log_T).to(alpha.dtype)
    contrib = torch.where(keep, T * alpha, 0.0)
    out = out.index_add(0, pix, contrib[:, None] * rgb)

    log_T_final = torch.zeros(H * W, dtype=L.dtype, device=L.device)
    log_T_final = log_T_final.index_add(0, pix, torch.where(keep, L, 0.0))
    T_final = torch.exp(log_T_final).to(alpha.dtype)
    if bg is not None:
        out = out + T_final[:, None] * bg.to(out)
    return out.view(H, W, D), (1.0 - T_final).view(H, W), info


def rasterize_dense(means2d: Tensor, conics: Tensor, radii: Tensor, colors: Tensor,
                    opacities: Tensor, H: int, W: int, depths: Tensor | None = None,
                    bg: Tensor | None = None, t_min: float = 1e-4,
                    alpha_min: float = 0.0) -> tuple[Tensor, Tensor]:
    """O(N * H * W) reference for :func:`rasterize` (alpha mode). Tiny scenes / tests only.

    Sorts Gaussians globally by depth and composites every Gaussian at every pixel with an
    explicit cumprod. Culled Gaussians (radius <= 0) are dropped.
    """
    order = torch.arange(means2d.shape[0], device=means2d.device) if depths is None \
        else torch.argsort(depths)
    order = order[radii[order] > 0]
    m, cn, c, o = means2d[order], conics[order], colors[order], opacities[order]
    ys, xs = torch.meshgrid(torch.arange(H, device=m.device), torch.arange(W, device=m.device),
                            indexing="ij")
    dx = xs.reshape(1, -1).to(m.dtype) + 0.5 - m[:, :1]                  # (N, HW)
    dy = ys.reshape(1, -1).to(m.dtype) + 0.5 - m[:, 1:]
    q = conic_quadratic(dx, dy, cn[:, None, :])
    alpha = torch.where(q < Q_CUTOFF, o[:, None] * torch.exp(-0.5 * q), 0.0).clamp(max=ALPHA_MAX)
    if alpha_min > 0:
        alpha = torch.where(alpha >= alpha_min, alpha, 0.0)
    one_minus = 1.0 - alpha
    T = torch.cumprod(torch.cat([torch.ones_like(one_minus[:1]), one_minus[:-1]]), dim=0)
    keep = (T * one_minus >= t_min) if t_min > 0 else torch.ones_like(T, dtype=torch.bool)
    img = torch.einsum("np,nd->pd", torch.where(keep, T * alpha, 0.0), c)
    T_final = torch.prod(torch.where(keep, one_minus, 1.0), dim=0)
    if bg is not None:
        img = img + T_final[:, None] * bg.to(img)
    return img.view(H, W, -1), (1.0 - T_final).view(H, W)
