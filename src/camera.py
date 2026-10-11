r"""Pinhole cameras.

Conventions (same as COLMAP, OpenCV and gsplat)
-----------------------------------------------
* Camera frame: x right, y down, z forward (the viewing direction). A point is in front
  of the camera iff z_cam > 0.
* ``viewmat`` is the 4x4 world-to-camera matrix acting on column vectors:
      [x_cam; 1] = viewmat @ [x_world; 1],   viewmat = [[R, t], [0, 1]],
  so x_cam = R x_world + t, the camera center is C = -R^T t, and the rows of R are the
  camera's x/y/z axes expressed in world coordinates. For row-stacked points (N, 3):
      X_cam = X_world @ R^T + t.
* Intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] map camera coordinates to
  continuous pixel coordinates  u = fx x/z + cx,  v = fy y/z + cy,  with the origin at
  the top-left *corner* of the image, so pixel (row i, col j) has its center at
  (j + 0.5, i + 0.5). (COLMAP uses this convention; for an ideal centered camera
  cx = W/2, cy = H/2.)
* Blender / NeRF-synthetic poses are camera-to-world in the OpenGL convention
  (x right, y up, z backward); they are converted by flipping the camera y and z axes
  (Stage 4 loader).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> Tensor:
    """(3, 3) float32 pinhole intrinsics."""
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def focal_from_fov(fov: float, size: int) -> float:
    """Focal length in pixels for a full field of view ``fov`` (radians) spanning ``size`` px."""
    return 0.5 * size / math.tan(0.5 * fov)


def look_at(eye: Tensor, target: Tensor, up: Tensor) -> Tensor:
    """World-to-camera (4, 4) for a camera at ``eye`` looking at ``target``, OpenCV axes.

    z_cam = normalize(target - eye); the image "up" (-y_cam) is as close to the world
    ``up`` vector as possible:
        x_cam = normalize(z_cam x up),   y_cam = z_cam x x_cam   (right-handed, y down).
    """
    z = torch.nn.functional.normalize(target - eye, dim=-1)
    x = torch.nn.functional.normalize(torch.linalg.cross(z, up), dim=-1)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], dim=0)                 # rows = camera axes in world coords
    V = torch.eye(4, dtype=eye.dtype, device=eye.device)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


@dataclass
class Camera:
    """One posed pinhole view. ``viewmat`` (4, 4) world-to-camera, ``K`` (3, 3)."""
    viewmat: Tensor
    K: Tensor
    width: int
    height: int
    name: str = ""

    @property
    def center(self) -> Tensor:
        """Camera center in world coordinates, C = -R^T t."""
        R, t = self.viewmat[:3, :3], self.viewmat[:3, 3]
        return -R.T @ t

    def to(self, *args, **kwargs) -> "Camera":
        return Camera(self.viewmat.to(*args, **kwargs), self.K.to(*args, **kwargs),
                      self.width, self.height, self.name)


def orbit_cameras(n: int, radius: float, elevation_deg: float, fov_deg: float, W: int, H: int,
                  target: Tensor | None = None, up: Tensor | None = None,
                  dtype=torch.float32, device="cpu") -> list[Camera]:
    """``n`` cameras evenly spaced in azimuth on a circle around ``target`` (world +y up)."""
    target = torch.zeros(3, dtype=dtype, device=device) if target is None else target
    up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device) if up is None else up
    f = focal_from_fov(math.radians(fov_deg), W)
    K = intrinsics_matrix(f, f, W / 2, H / 2).to(dtype=dtype, device=device)
    el = math.radians(elevation_deg)
    cams = []
    for i in range(n):
        az = 2 * math.pi * i / n
        eye = target + radius * torch.tensor(
            [math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)],
            dtype=dtype, device=device)
        cams.append(Camera(look_at(eye, target, up), K.clone(), W, H, name=f"orbit_{i:03d}"))
    return cams


def sphere_cameras(n: int, radius: float, fov_deg: float, W: int, H: int,
                   target: Tensor | None = None, seed: int = 0,
                   dtype=torch.float32, device="cpu") -> list[Camera]:
    """``n`` cameras on a sphere around ``target`` (Fibonacci lattice, randomly rotated by
    ``seed`` about the y axis), all looking at it. Near the poles, where the world +y
    "up" is parallel to the viewing direction, +z is used as the up hint instead."""
    target = torch.zeros(3, dtype=dtype, device=device) if target is None else target
    f = focal_from_fov(math.radians(fov_deg), W)
    K = intrinsics_matrix(f, f, W / 2, H / 2).to(dtype=dtype, device=device)
    phase = 2 * math.pi * torch.rand(1, generator=torch.Generator().manual_seed(seed)).item()
    golden = math.pi * (3 - math.sqrt(5))
    cams = []
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(1 - y * y)
        a = golden * i + phase
        d = torch.tensor([r * math.cos(a), y, r * math.sin(a)], dtype=dtype, device=device)
        up = torch.tensor([0.0, 1.0, 0.0] if abs(y) < 0.95 else [0.0, 0.0, 1.0],
                          dtype=dtype, device=device)
        cams.append(Camera(look_at(target + radius * d, target, up), K.clone(), W, H,
                           name=f"sphere_{i:03d}"))
    return cams


# Blender's camera axes (x right, y up, z backward) -> OpenCV (x right, y down, z forward):
# negate the camera's y and z axes, i.e. right-multiply camera-to-world by diag(1, -1, -1, 1).
OPENGL_TO_OPENCV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def load_blender(scene_dir, split: str, downscale: int = 1, bg=(1.0, 1.0, 1.0)
                 ) -> tuple[list[Camera], Tensor]:
    """NeRF-synthetic split -> (cameras, images uint8 (N, H, W, 3)).

    ``transforms_<split>.json`` holds ``camera_angle_x`` (full horizontal FOV, radians)
    and per-frame ``transform_matrix``: camera-to-world in the OpenGL convention.
    viewmat = inverse(c2w_gl @ OPENGL_TO_OPENCV). Images are RGBA; they are composited
    onto ``bg`` at full resolution (C = a rgb + (1 - a) bg) and then area-downsampled by
    ``downscale`` (compositing and box filtering are both linear, so the order is exact
    for box filtering). Intrinsics: f = W / (2 tan(fov/2)), principal point at the center.
    """
    import json
    from pathlib import Path

    import cv2
    import numpy as np

    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / f"transforms_{split}.json").read_text())
    bg = np.asarray(bg, np.float32)
    cams, imgs = [], []
    for fr in meta["frames"]:
        path = scene_dir / (fr["file_path"].removeprefix("./") + ".png")
        rgba = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        rgba = rgba.astype(np.float32) / 255.0
        rgb = rgba[..., :3] * rgba[..., 3:] + (1.0 - rgba[..., 3:]) * bg
        H0, W0 = rgb.shape[:2]
        W, H = W0 // downscale, H0 // downscale
        if downscale > 1:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        imgs.append(torch.from_numpy(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)))
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float64) @ OPENGL_TO_OPENCV.double()
        f = focal_from_fov(meta["camera_angle_x"], W)
        cams.append(Camera(torch.linalg.inv(c2w).float(), intrinsics_matrix(f, f, W / 2, H / 2),
                           W, H, name=Path(fr["file_path"]).name))
    return cams, torch.stack(imgs)
