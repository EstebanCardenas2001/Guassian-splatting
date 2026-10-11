r"""Reference 3D Gaussian Splatting renderer in pure PyTorch (Stage 2).

Pipeline for one camera (conventions in ``src/camera.py``):

1. World -> camera:   t_k = R mu_k + t                      (viewmat = [[R, t], [0, 1]])
2. Culling:           near < t_z < far, finite positive-definite 2D covariance, and the
                      3-sigma box overlaps the image.
3. Pinhole:           u = fx t_x / t_z + cx,  v = fy t_y / t_z + cy
4. EWA (Zwicker et al. 2002): the perspective map is replaced by its first-order
   Taylor expansion at t_k, so a 3D Gaussian maps to a 2D Gaussian with
       Sigma'_k = J_k R Sigma_k R^T J_k^T + eps I,
       J_k = d(u, v)/d(t) = [[fx/t_z, 0, -fx t_x/t_z^2],
                             [0, fy/t_z, -fy t_y/t_z^2]]          (2 x 3)
   R Sigma R^T rotates the covariance into the camera frame (W in the paper's notation).
   eps = 0.3 px^2 is a low-pass "dilation" that keeps every splat at least ~1 px wide,
   for numerical stability and crude anti-aliasing. As in 3DGS/gsplat, t_x/t_z and
   t_y/t_z are clamped to 1.3x the field of view before building J, so Gaussians far
   outside the frustum don't get absurdly stretched footprints.
5. Composite the 2D Gaussians front to back by camera depth t_z (``src/rasterize.py``):
       C = sum_k c_k alpha_k prod_{j<k} (1 - alpha_j) + T_final bg,
       alpha_k = min(0.99, o_k exp(-q_k / 2)),   q_k = d^T Sigma'^{-1}_k d.

Sorting is per pixel by the depth of each Gaussian's *center*, like 3DGS: it is exact
only for non-overlapping Gaussians, which is the known source of "popping" artifacts.
"""
from __future__ import annotations

import torch
from torch import Tensor

from src.gaussians import covariance_3d
from src.rasterize import rasterize


def world_to_camera(means: Tensor, viewmat: Tensor) -> Tensor:
    """(N, 3) world points -> (N, 3) camera coordinates: X @ R^T + t."""
    return means @ viewmat[:3, :3].T + viewmat[:3, 3]


def project_points(p_cam: Tensor, K: Tensor) -> Tensor:
    """(N, 3) camera coordinates -> (N, 2) pixel coordinates (u, v). Requires z != 0."""
    z = p_cam[:, 2]
    return torch.stack([K[0, 0] * p_cam[:, 0] / z + K[0, 2],
                        K[1, 1] * p_cam[:, 1] / z + K[1, 2]], dim=-1)


def ewa_covariance_2d(cov3d: Tensor, p_cam: Tensor, viewmat: Tensor, K: Tensor, W: int, H: int,
                      eps2d: float = 0.3) -> Tensor:
    """Screen-space covariance Sigma' = J R Sigma R^T J^T + eps2d I.  (N, 2, 2) in px^2.

    cov3d (N, 3, 3) world frame; p_cam (N, 3) camera-frame means with z > 0.
    The clamp limits follow gsplat: |t_x/t_z| <= (distance to the image border)/fx
    + 0.3 tan(fov_x / 2), and likewise for y.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = p_cam.unbind(-1)
    tan_fx, tan_fy = 0.5 * W / fx, 0.5 * H / fy
    tx = z * torch.minimum(torch.maximum(x / z, -(cx / fx + 0.3 * tan_fx)), (W - cx) / fx + 0.3 * tan_fx)
    ty = z * torch.minimum(torch.maximum(y / z, -(cy / fy + 0.3 * tan_fy)), (H - cy) / fy + 0.3 * tan_fy)
    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zero, -fx * tx / (z * z)], -1),
                     torch.stack([zero, fy / z, -fy * ty / (z * z)], -1)], dim=-2)   # (N, 2, 3)
    R = viewmat[:3, :3]
    T = J @ R                                                                          # (N, 2, 3)
    cov2d = T @ cov3d @ T.transpose(-1, -2)
    return cov2d + eps2d * torch.eye(2, dtype=cov2d.dtype, device=cov2d.device)


def project_gaussians(means: Tensor, quats: Tensor, scales: Tensor, viewmat: Tensor, K: Tensor,
                      W: int, H: int, near: float = 0.01, far: float = 1e10, eps2d: float = 0.3,
                      antialiased: bool = False, q_cutoff: float = 9.0) -> dict[str, Tensor]:
    """Project N 3D Gaussians to the image plane.

    Returns a dict of per-Gaussian tensors:
        means2d (N, 2) px, cov2d (N, 2, 2), conics (N, 3) = (A, B, C) of cov2d^{-1},
        depths (N,) camera z, radii (N,) extent sqrt(q_cutoff * lambda_max) in px (3 sigma for
        the default q_cutoff = 9; 0 if culled), valid (N,) bool,
        opacity_scale (N,): 1, or sqrt(det(cov2d - eps I) / det(cov2d)) in antialiased mode
        (Mip-Splatting's 2D filter: dilation that *conserves* each splat's total energy
        instead of inflating sub-pixel Gaussians).
    Culled Gaussians get safe placeholder values so no inf/NaN can leak into gradients.
    """
    p_cam = world_to_camera(means, viewmat)
    depths = p_cam[:, 2]
    in_depth = (depths > near) & (depths < far)
    # Placeholder depth for culled Gaussians: avoids 0 * inf = NaN in the backward pass.
    p_safe = torch.where(in_depth[:, None], p_cam, torch.tensor([0.0, 0.0, 1.0]).to(p_cam))

    cov2d = ewa_covariance_2d(covariance_3d(quats, scales), p_safe, viewmat, K, W, H, eps2d)
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    ok = in_depth & (det > 0)
    det_safe = torch.where(ok, det, 1.0)
    conics = torch.stack([c, -b, a], dim=-1) / det_safe[:, None]
    means2d = project_points(p_safe, K)

    with torch.no_grad():
        mid = 0.5 * (a + c)
        lam_max = mid + torch.sqrt((mid * mid - det).clamp(min=0.0))
        radius = torch.sqrt(q_cutoff * lam_max.clamp(min=0.0))
        u, v = means2d[:, 0], means2d[:, 1]
        on_screen = (u + radius > 0) & (u - radius < W) & (v + radius > 0) & (v - radius < H)
        valid = ok & on_screen & torch.isfinite(means2d).all(-1)
        radii = torch.where(valid, radius, 0.0)

    if antialiased:
        det_raw = (a - eps2d) * (c - eps2d) - b * b
        opacity_scale = torch.sqrt((det_raw / det_safe).clamp(min=0.0))
    else:
        opacity_scale = torch.ones_like(det)
    return {"means2d": means2d, "cov2d": cov2d, "conics": conics, "depths": depths,
            "radii": radii, "valid": valid, "opacity_scale": opacity_scale}


def render(means: Tensor, quats: Tensor, scales: Tensor, opacities: Tensor, colors: Tensor,
           viewmat: Tensor, K: Tensor, W: int, H: int, bg: Tensor | None = None,
           near: float = 0.01, far: float = 1e10, eps2d: float = 0.3, antialiased: bool = False,
           t_min: float = 1e-4, alpha_min: float = 1 / 255,
           max_pairs: int = 80_000_000, alpha_max: float = 0.99,
           q_cutoff: float = 9.0) -> tuple[Tensor, Tensor, dict]:
    """Render N activated 3D Gaussians from one camera.

    Args:
        means (N, 3) world, quats (N, 4) wxyz (any norm), scales (N, 3) > 0,
        opacities (N,) in (0, 1), colors (N, D), viewmat (4, 4) world-to-camera, K (3, 3).
        bg: (D,) background color (None = black).
        t_min: early-termination transmittance; alpha_min: skip splats with alpha below it
            (3DGS uses 1/255).
        alpha_max, q_cutoff: alpha clamp and support q < q_cutoff. Defaults follow the 3DGS
            paper (0.99, 3 sigma); gsplat 1.5 uses 0.999 and 3.33 sigma (q < 11.09).
    Returns:
        image (H, W, D), alpha (H, W) = accumulated opacity, info dict with the projection
        outputs (``means2d`` is the tensor whose gradient drives densification in Stage 7)
        and ``pairs``.
    """
    proj = project_gaussians(means, quats, scales, viewmat, K, W, H, near, far, eps2d, antialiased,
                             q_cutoff)
    opac = opacities * proj["opacity_scale"]
    radii, q_max = proj["radii"], None
    if alpha_min > 0:
        # Opacity-aware support (as in gsplat): alpha = o exp(-q/2) >= alpha_min  <=>
        # q <= 2 ln(o / alpha_min). Pairs beyond it would be skipped anyway, so the image is
        # unchanged; it only avoids enumerating them. Shrinks the pixel box accordingly.
        with torch.no_grad():
            q_max = (2 * torch.log(opac.detach() / alpha_min)).clamp(max=q_cutoff)
            radii = torch.where(q_max > 0, radii * torch.sqrt(q_max.clamp(min=0) / q_cutoff), 0.0)
    img, alpha, info = rasterize(proj["means2d"], proj["conics"], radii, colors, opac, H, W,
                                 depths=proj["depths"], bg=bg, t_min=t_min, alpha_min=alpha_min,
                                 q_max=q_max if q_max is not None else torch.full_like(opac, q_cutoff),
                                 max_pairs=max_pairs, alpha_max=alpha_max)
    info.update(proj)
    return img, alpha, info
