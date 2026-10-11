r"""3D Gaussian parameters, activations and covariance.

A 3D Gaussian k has (world frame)
    mean      mu_k in R^3
    rotation  unit quaternion q_k = (w, x, y, z)   (stored unnormalized, normalized on use)
    scales    s_k = exp(log_scales_k) in R^3_{>0}   (std. dev. along the rotated axes)
    opacity   o_k = sigmoid(opacity_logit_k)
    color     c_k = sigmoid(color_logit_k) in [0, 1]^3   (spherical harmonics in Stage 7D)

Covariance:  Sigma = R S S^T R^T = M M^T,  M = R S,  S = diag(s).
This factorization keeps Sigma symmetric positive semi-definite under any gradient step,
which optimizing Sigma's entries directly would not.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def quat_to_rotmat(q: Tensor) -> Tensor:
    """(..., 4) quaternions (w, x, y, z), any norm  ->  (..., 3, 3) rotation matrices.

    R acts on column vectors (v' = R v). q and -q, and any positive rescaling of q,
    give the same R.
    """
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def covariance_3d(quats: Tensor, scales: Tensor) -> Tensor:
    """Sigma = R diag(s)^2 R^T for quats (N, 4) and scales (N, 3)  ->  (N, 3, 3)."""
    M = quat_to_rotmat(quats) * scales[..., None, :]     # R S: column j of R times s_j
    return M @ M.transpose(-1, -2)


def logit(p: Tensor, eps: float = 1e-6) -> Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


class Gaussians(nn.Module):
    """Raw (unconstrained) parameters of N 3D Gaussians; call :meth:`activated` to render."""

    def __init__(self, means: Tensor, quats: Tensor, log_scales: Tensor,
                 opacity_logits: Tensor, color_logits: Tensor):
        super().__init__()
        self.means = nn.Parameter(means)
        self.quats = nn.Parameter(quats)
        self.log_scales = nn.Parameter(log_scales)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.color_logits = nn.Parameter(color_logits)

    def __len__(self) -> int:
        return self.means.shape[0]

    def activated(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """(means, quats, scales, opacities, colors) in the order ``render`` expects."""
        return (self.means, self.quats, torch.exp(self.log_scales),
                torch.sigmoid(self.opacity_logits), torch.sigmoid(self.color_logits))

    @classmethod
    def from_activated(cls, means, quats, scales, opacities, colors) -> "Gaussians":
        return cls(means.clone(), quats.clone(), torch.log(scales), logit(opacities),
                   logit(colors))

    @classmethod
    def random_box(cls, n: int, lo: Tensor, hi: Tensor, scale: float | None = None,
                   opacity: float = 0.1, generator: torch.Generator | None = None) -> "Gaussians":
        """Uniform means in the box [lo, hi], random rotations, isotropic scale
        (default: mean spacing (volume / n)^(1/3)), constant opacity, random colors."""
        dev = lo.device
        means = lo + (hi - lo) * torch.rand(n, 3, generator=generator, device=dev)
        if scale is None:
            scale = float(torch.prod(hi - lo)) ** (1 / 3) / n ** (1 / 3)
        quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator, device=dev), dim=-1)
        return cls(means, quats, torch.full((n, 3), math.log(scale), device=dev),
                   logit(torch.full((n,), opacity, device=dev)),
                   logit(torch.rand(n, 3, generator=generator, device=dev)))
