"""Blender loader conventions (needs data/nerf_synthetic/lego; skipped otherwise)."""
import math
from pathlib import Path

import pytest
import torch

from src.camera import load_blender

LEGO = Path(__file__).resolve().parents[1] / "data" / "nerf_synthetic" / "lego"
pytestmark = pytest.mark.skipif(not LEGO.exists(), reason="lego not downloaded")


@pytest.fixture(scope="module")
def lego():
    return load_blender(LEGO, "train", downscale=8)


def test_shapes_and_intrinsics(lego):
    cams, imgs = lego
    assert imgs.shape == (100, 100, 100, 3) and imgs.dtype == torch.uint8
    c = cams[0]
    assert (c.width, c.height) == (100, 100)
    assert abs(c.K[0, 0].item() - 100 / (2 * math.tan(0.6911112 / 2))) < 1e-3


def test_cameras_are_rigid_and_look_at_the_origin(lego):
    """All lego cameras sit on a sphere (r ~ 4.03) aimed at the origin: in OpenCV
    convention the origin must be in front (z > 0) and project near the image center."""
    cams, _ = lego
    for c in cams:
        R = c.viewmat[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert torch.isclose(torch.linalg.det(R), torch.tensor(1.0), atol=1e-5)
        assert abs(c.center.norm().item() - 4.03) < 0.05
        o = c.viewmat[:3, 3]                       # origin in camera coordinates
        assert o[2] > 3.9
        u = c.K[0, 0] * o[0] / o[2] + c.K[0, 2]
        v = c.K[1, 1] * o[1] / o[2] + c.K[1, 2]
        assert abs(u - 50) < 1.0 and abs(v - 50) < 1.0


def test_world_up_is_image_up(lego):
    """Blender's world is z-up: a point above the origin must appear above the image center."""
    cams, _ = lego
    for c in cams:
        p = c.viewmat[:3, :3] @ torch.tensor([0.0, 0.0, 0.5]) + c.viewmat[:3, 3]
        assert p[1] / p[2] < 0                     # negative camera y = upper half of image
