r"""gsplat's tiled CUDA rasterizer behind the same interface as ``render_torch.render`` (Stage 5).

Same conventions (OpenCV camera, world-to-camera ``viewmat`` on column vectors, wxyz
quaternions, activated parameters) and the same EWA model, so a trained model can switch
renderer freely. What gsplat does differently, read from its 1.5.3 CUDA source:

* **Tiles.** Each Gaussian is assigned to the 16x16 tiles overlapped by its bounding box;
  (tile, Gaussian) pairs are sorted once by (tile, depth) with a radix sort; one CUDA
  block per tile walks its list front to back in shared memory and **stops** once every
  pixel of the tile is saturated. Hidden Gaussians are never evaluated, unlike the
  pair-based PyTorch renderer, which builds and differentiates every pair.
* **Support:** opacity-aware box of half-width min(3.33, sqrt(2 ln(255 o))) * sqrt(Sigma'_ii)
  per axis, no ellipse test; every pixel of a covered tile with alpha >= 1/255 counts.
  (render_torch: ellipse q < min(9, 2 ln(255 o)); equal to gsplat with q_cutoff = 3.33^2
  except for o > 0.98, where gsplat also shades pixels beyond 3.33 sigma inside covered tiles.)
* **alpha = min(0.999, o G)** (render_torch default 0.99, as in the 3DGS paper).
* Early termination at T <= 1e-4 and the 1/255 threshold are compiled in, so ``t_min`` and
  ``alpha_min`` are accepted for interface compatibility but must keep their defaults.
"""
from __future__ import annotations

import torch
from torch import Tensor

GSPLAT_ALPHA_MAX = 0.999
GSPLAT_Q_CUTOFF = 3.33 ** 2


def render(means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
           viewmat: Tensor, K: Tensor, W: int, H: int, bg: Tensor | None = None,
           near: float = 0.01, far: float = 1e10, eps2d: float = 0.3, antialiased: bool = False,
           t_min: float = 1e-4, alpha_min: float = 1 / 255, max_pairs: int | None = None,
           packed: bool = True, sparse_grad: bool = False, absgrad: bool = False
           ) -> tuple[Tensor, Tensor, dict]:
    """Render N activated 3D Gaussians from one camera with gsplat (float32, CUDA).

    Args/returns as :func:`src.render_torch.render`. Extra: ``packed`` (only visible
    Gaussians are carried through the pipeline - less memory), ``sparse_grad``,
    ``absgrad`` (also accumulate |dL/dmeans2d| per pixel, for AbsGS densification).

    info: ``means2d`` (gsplat's projected means; call ``retain_grad()`` before backward to
    read the screen-space gradient), ``radii`` (N,) max per-axis extent in px (0 = culled),
    ``depths`` (N,), ``pairs`` = number of (tile, Gaussian) intersections, ``meta`` (raw).
    """
    from gsplat import rasterization

    if t_min != 1e-4 or alpha_min != 1 / 255:
        raise ValueError("gsplat compiles t_min = 1e-4 and alpha_min = 1/255 in")
    rc, ra, meta = rasterization(
        means, quats, scales, opacities, colors, viewmat[None], K[None], W, H,
        near_plane=near, far_plane=far, eps2d=eps2d, packed=packed,
        rasterize_mode="antialiased" if antialiased else "classic",
        sparse_grad=sparse_grad, absgrad=absgrad)
    img, alpha = rc[0], ra[0, ..., 0]
    if bg is not None:
        # Composite here rather than via gsplat's ``backgrounds``: in 1.5.3 packed mode its
        # shape check derives image dims from the packed means2d and rejects any background.
        # C + (1 - alpha) bg is exactly the T_final * bg term, and differentiable via alpha.
        img = img + (1.0 - alpha)[..., None] * bg

    N = means.shape[0]
    with torch.no_grad():
        r = meta["radii"].amax(-1).to(means.dtype)                 # packed: (nnz,), else (1, N)
        radii = torch.zeros(N, dtype=means.dtype, device=means.device)
        depths = torch.zeros(N, dtype=means.dtype, device=means.device)
        if packed:
            radii[meta["gaussian_ids"]] = r
            depths[meta["gaussian_ids"]] = meta["depths"].to(means.dtype)
        else:
            radii, depths = r[0], meta["depths"][0].to(means.dtype)
    info = {"means2d": meta["means2d"], "radii": radii, "depths": depths,
            "valid": radii > 0, "pairs": int(meta["flatten_ids"].numel()), "meta": meta}
    return img, alpha, info
