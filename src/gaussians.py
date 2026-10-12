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
    """Raw (unconstrained) parameters of N 3D Gaussians.

    Stored in ``self.params``, an ``nn.ParameterDict`` keyed with gsplat's names so that
    densification (mine or gsplat's) can grow/prune every tensor in place:
        means (N, 3) | quats (N, 4) | scales (N, 3) = log-scales | opacities (N,) = logits
    and one color representation:
        colors (N, 3) = RGB logits, c = sigmoid(.)                        (view-independent)
     or sh0 (N, 1, 3) + shN (N, (L+1)^2 - 1, 3) = spherical-harmonics coefficients (src/sh.py)
    The Stage 3-4 names (``log_scales``, ``opacity_logits``, ``color_logits``) are aliases.
    """

    KEYS = ("means", "quats", "scales", "opacities", "colors")
    ALIASES = {"log_scales": "scales", "opacity_logits": "opacities", "color_logits": "colors"}

    def __init__(self, means: Tensor, quats: Tensor, log_scales: Tensor, opacity_logits: Tensor,
                 color_logits: Tensor | None = None, sh0: Tensor | None = None,
                 shN: Tensor | None = None):
        super().__init__()
        d = {"means": nn.Parameter(means), "quats": nn.Parameter(quats),
             "scales": nn.Parameter(log_scales), "opacities": nn.Parameter(opacity_logits)}
        if color_logits is not None:
            d["colors"] = nn.Parameter(color_logits)
        else:
            d["sh0"], d["shN"] = nn.Parameter(sh0), nn.Parameter(shN)
        self.params = nn.ParameterDict(d)

    def __getattr__(self, name):
        params = self._modules["params"]
        if name in Gaussians.KEYS and name in params:
            return params[name]
        if name in Gaussians.ALIASES:
            return params[Gaussians.ALIASES[name]]
        return super().__getattr__(name)

    def __len__(self) -> int:
        return self.params["means"].shape[0]

    @property
    def uses_sh(self) -> bool:
        return "sh0" in self.params

    @property
    def sh_degree(self) -> int:
        """Maximum SH degree L stored: (L+1)^2 = 1 + shN.shape[1]."""
        return round(math.sqrt(1 + self.params["shN"].shape[1])) - 1 if self.uses_sh else 0

    def geometry(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(means, quats, scales, opacities), activated."""
        p = self.params
        return p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"])

    def colors_for(self, viewmat: Tensor, degree: int | None = None) -> Tensor:
        """(N, 3) colors seen from the camera ``viewmat`` (SH evaluated up to ``degree``)."""
        if not self.uses_sh:
            return torch.sigmoid(self.params["colors"])
        from src.sh import sh_to_color, view_dirs
        coeffs = torch.cat([self.params["sh0"], self.params["shN"]], dim=1)
        deg = self.sh_degree if degree is None else min(degree, self.sh_degree)
        return sh_to_color(coeffs, view_dirs(self.params["means"].detach(), viewmat), deg)

    def activated(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """(means, quats, scales, opacities, colors) for view-independent RGB models."""
        if self.uses_sh:
            raise ValueError("SH colors depend on the camera: use geometry() + colors_for()")
        return (*self.geometry(), torch.sigmoid(self.params["colors"]))

    @classmethod
    def from_state_dict(cls, sd: dict) -> "Gaussians":
        """Rebuild from a state dict of any size (densification changes N), in the current
        layout (``params.<key>``) or the Stage 3-4 layout (``means``, ``log_scales``...)."""
        def get(key):
            for k in (f"params.{key}", key, *[a for a, v in cls.ALIASES.items() if v == key]):
                if k in sd:
                    return sd[k].clone()
            return None
        geo = [get(k) for k in ("means", "quats", "scales", "opacities")]
        if get("colors") is not None:
            return cls(*geo, color_logits=get("colors"))
        return cls(*geo, sh0=get("sh0"), shN=get("shN"))

    @classmethod
    def from_activated(cls, means, quats, scales, opacities, colors) -> "Gaussians":
        return cls(means.clone(), quats.clone(), torch.log(scales), logit(opacities),
                   logit(colors))

    @classmethod
    def from_points(cls, xyz: Tensor, rgb: Tensor, sh_degree: int | None = None,
                    opacity: float = 0.1) -> "Gaussians":
        """3DGS initialization from a point cloud (e.g. SfM): one isotropic Gaussian per
        point with scale = RMS distance to its 3 nearest neighbours, identity rotation,
        constant opacity, the point's color (RGB logits, or SH degree 0 with zero higher
        coefficients when ``sh_degree`` is given). xyz (N, 3), rgb (N, 3) in [0, 1]."""
        from scipy.spatial import cKDTree
        pts = xyz.detach().cpu().double().numpy()
        d, _ = cKDTree(pts).query(pts, k=4)                   # self + 3 neighbours
        scale = torch.from_numpy((d[:, 1:] ** 2).mean(1).clip(min=1e-14) ** 0.5).float()
        dev = xyz.device
        n = len(xyz)
        quats = torch.zeros(n, 4, device=dev)
        quats[:, 0] = 1.0
        log_s = torch.log(scale).to(dev)[:, None].repeat(1, 3)
        op = logit(torch.full((n,), opacity, device=dev))
        if sh_degree is None:
            return cls(xyz.clone(), quats, log_s, op, color_logits=logit(rgb.clamp(0.02, 0.98)))
        from src.sh import num_coeffs, rgb_to_sh0
        return cls(xyz.clone(), quats, log_s, op, sh0=rgb_to_sh0(rgb)[:, None, :],
                   shN=torch.zeros(n, num_coeffs(sh_degree) - 1, 3, device=dev))

    @classmethod
    def random_box(cls, n: int, lo: Tensor, hi: Tensor, scale: float | None = None,
                   opacity: float = 0.1, generator: torch.Generator | None = None,
                   sh_degree: int | None = None) -> "Gaussians":
        """Uniform means in the box [lo, hi], random rotations, isotropic scale
        (default: mean spacing (volume / n)^(1/3)), constant opacity, random colors."""
        dev = lo.device
        means = lo + (hi - lo) * torch.rand(n, 3, generator=generator, device=dev)
        if scale is None:
            scale = float(torch.prod(hi - lo)) ** (1 / 3) / n ** (1 / 3)
        quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator, device=dev), dim=-1)
        log_s = torch.full((n, 3), math.log(scale), device=dev)
        op = logit(torch.full((n,), opacity, device=dev))
        rgb = torch.rand(n, 3, generator=generator, device=dev)
        if sh_degree is None:
            return cls(means, quats, log_s, op, color_logits=logit(rgb))
        from src.sh import num_coeffs, rgb_to_sh0
        return cls(means, quats, log_s, op, sh0=rgb_to_sh0(rgb)[:, None, :],
                   shN=torch.zeros(n, num_coeffs(sh_degree) - 1, 3, device=dev))
