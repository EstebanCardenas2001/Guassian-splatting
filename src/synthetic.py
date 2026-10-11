r"""Stage 3: synthetic ground-truth scenes and parameter-recovery metrics.

Identifiability of the parameters
---------------------------------
Even with perfect images, the raw parameters (mu, q, s, o, c) are not unique:

* **Label switching.** Permuting the Gaussians changes nothing, so recovered and
  ground-truth Gaussians are matched with the Hungarian algorithm before comparing.
* **Quaternion double cover.** q and -q give the same rotation.
* **Axis permutation and flips.** Sigma = R S^2 R^T is unchanged if two columns of R are
  swapped together with the corresponding scales, or a column is negated. Only Sigma is
  identifiable, not (q, s), so the comparison uses Sigma (and its eigen-decomposition).
* **Isotropy.** If two scales are equal, the rotation about the remaining axis is
  arbitrary; if all three are equal, R is entirely arbitrary. The principal-axis error is
  therefore only meaningful for clearly anisotropic Gaussians.
* **Photometric degeneracies** (not fixable by matching): a Gaussian whose color equals
  the background is invisible at any opacity; one hidden behind others in every view
  is unconstrained; with additive-looking overlap, color and opacity trade off (only the
  product c * alpha * T is observed when T is small).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.gaussians import covariance_3d


def make_gt_scene(n: int, extent: float = 1.0, scale_range=(0.04, 0.15),
                  opacity_range=(0.6, 0.95), seed: int = 0, device="cpu") -> dict[str, Tensor]:
    """n Gaussians with means uniform in [-extent, extent]^3, log-uniform anisotropic scales,
    uniformly random rotations (normalized Gaussian 4-vectors), random opacities and colors."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = math.log(scale_range[0]), math.log(scale_range[1])
    scene = {
        "means": (2 * torch.rand(n, 3, generator=g) - 1) * extent,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "scales": torch.exp(lo + (hi - lo) * torch.rand(n, 3, generator=g)),
        "opacities": opacity_range[0] + (opacity_range[1] - opacity_range[0]) * torch.rand(n, generator=g),
        "colors": torch.rand(n, 3, generator=g),
    }
    return {k: v.to(device) for k, v in scene.items()}


def _principal_axis(cov: Tensor) -> tuple[Tensor, Tensor]:
    """Unit eigenvector of the largest eigenvalue and the anisotropy ratio s_max / s_mid."""
    ev, vec = torch.linalg.eigh(cov)                       # ascending
    return vec[..., :, 2], torch.sqrt(ev[..., 2] / ev[..., 1].clamp(min=1e-20))


@torch.no_grad()
def compare_parameters(gt: dict[str, Tensor], rec: dict[str, Tensor], pos_tol: float = 0.03,
                       cov_tol: float = 0.25, color_tol: float = 0.1, opacity_tol: float = 0.1
                       ) -> dict:
    """Match recovered to ground-truth Gaussians (Hungarian on mean distance) and compare.

    ``rec`` may contain more Gaussians than ``gt`` (unmatched ones are ignored).
    Returns per-GT-Gaussian arrays (numpy, in GT order) and summary scalars:
        pos_err           |mu_rec - mu_gt|
        cov_rel_err       ||Sigma_rec - Sigma_gt||_F / ||Sigma_gt||_F
        axis_err_deg      angle between largest principal axes (sign-agnostic)
        anisotropy        s_max / s_mid of the GT Gaussian (axis error meaningful when >> 1)
        naive_quat_err_deg  rotation angle between q_rec and q_gt, ignoring the equivalences
        opacity_err, color_err (RMS over channels)
        recovered_frac    fraction meeting all four tolerances
    """
    gt = {k: v.detach().double().cpu() for k, v in gt.items()}
    rec = {k: v.detach().double().cpu() for k, v in rec.items()}
    cost = torch.cdist(gt["means"], rec["means"]).numpy()
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(len(gt["means"]), dtype=np.int64)
    order[rows] = cols
    r = {k: v[torch.from_numpy(order)] for k, v in rec.items()}

    pos_err = (r["means"] - gt["means"]).norm(dim=-1)
    S_gt = covariance_3d(gt["quats"], gt["scales"])
    S_rec = covariance_3d(r["quats"], r["scales"])
    cov_rel = (S_rec - S_gt).flatten(1).norm(dim=-1) / S_gt.flatten(1).norm(dim=-1)
    a_gt, aniso = _principal_axis(S_gt)
    a_rec, _ = _principal_axis(S_rec)
    axis_err = torch.rad2deg(torch.arccos((a_gt * a_rec).sum(-1).abs().clamp(max=1.0)))
    qg = gt["quats"] / gt["quats"].norm(dim=-1, keepdim=True)
    qr = r["quats"] / r["quats"].norm(dim=-1, keepdim=True)
    naive = torch.rad2deg(2 * torch.arccos((qg * qr).sum(-1).clamp(-1.0, 1.0)))
    op_err = (r["opacities"] - gt["opacities"]).abs()
    col_err = (r["colors"] - gt["colors"]).pow(2).mean(-1).sqrt()
    ok = (pos_err < pos_tol) & (cov_rel < cov_tol) & (col_err < color_tol) & (op_err < opacity_tol)
    return {
        "match": order,
        "pos_err": pos_err.numpy(), "cov_rel_err": cov_rel.numpy(),
        "axis_err_deg": axis_err.numpy(), "anisotropy": aniso.numpy(),
        "naive_quat_err_deg": naive.numpy(),
        "opacity_err": op_err.numpy(), "color_err": col_err.numpy(),
        "recovered": ok.numpy(), "recovered_frac": float(ok.double().mean()),
    }


def summarize(m: dict) -> dict:
    """Scalar summary of :func:`compare_parameters` for metrics.json."""
    aniso = m["anisotropy"] > 1.5
    return {
        "recovered_frac": m["recovered_frac"],
        "pos_err_median": float(np.median(m["pos_err"])),
        "cov_rel_err_median": float(np.median(m["cov_rel_err"])),
        "axis_err_deg_median_anisotropic": float(np.median(m["axis_err_deg"][aniso])) if aniso.any() else None,
        "naive_quat_err_deg_median": float(np.median(m["naive_quat_err_deg"])),
        "opacity_err_median": float(np.median(m["opacity_err"])),
        "color_err_median": float(np.median(m["color_err"])),
    }
