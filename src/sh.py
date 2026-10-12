r"""Real spherical harmonics for view-dependent color (Stage 7D).

Each Gaussian stores K = (L+1)^2 coefficient vectors k_lm in R^3 (one per color channel).
The color seen along the unit viewing direction d (from the camera center towards the
Gaussian mean, world frame) is

    c(d) = max(0, sum_{l<=L, |m|<=l} k_lm Y_lm(d) + 1/2)

with the real, orthonormal SH basis  int_{S^2} Y_lm Y_l'm' dOmega = delta_ll' delta_mm'.
Degree 0 is a constant color: Y_00 = 1/(2 sqrt(pi)) = C0, so k_00 = (c - 1/2) / C0.
Degrees 1-3 add 3, 5, 7 terms: low-frequency view dependence (sheen, soft highlights).
Coefficient order and constants follow 3DGS / gsplat (x, y, z are the components of d).
"""
from __future__ import annotations

import torch
from torch import Tensor

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
      0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def num_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def sh_basis(dirs: Tensor, degree: int) -> Tensor:
    """Y_lm(d) for unit directions (..., 3) -> (..., (degree+1)^2)."""
    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]
    if degree >= 1:
        out += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if degree >= 3:
        out += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy),
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return torch.stack(out, -1)


def sh_to_color(coeffs: Tensor, dirs: Tensor, degree: int) -> Tensor:
    """coeffs (N, K, 3) with K >= (degree+1)^2, dirs (N, 3) unit -> colors (N, 3) >= 0.

    Only the first (degree+1)^2 coefficients are used (progressive SH training)."""
    k = num_coeffs(degree)
    Y = sh_basis(dirs, degree)                                   # (N, k)
    return torch.clamp_min((Y[..., None] * coeffs[:, :k]).sum(-2) + 0.5, 0.0)


def view_dirs(means: Tensor, viewmat: Tensor) -> Tensor:
    """Unit directions from the camera center C = -R^T t to each mean, world frame. (N, 3)"""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    return torch.nn.functional.normalize(means - (-R.T @ t), dim=-1)


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    """Degree-0 coefficient that reproduces a constant color: k_00 = (c - 1/2) / C0."""
    return (rgb - 0.5) / C0
