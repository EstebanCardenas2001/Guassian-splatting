"""Stage 6: COLMAP loader conventions, checked against COLMAP's own observations.
Skipped when the plant reconstruction isn't present."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.camera import load_colmap
from src.render_torch import project_points, world_to_camera

SCENE = Path(__file__).resolve().parents[1] / "data" / "plant"
MODEL = "colmap_incremental/sparse/1"
pytestmark = pytest.mark.skipif(not (SCENE / MODEL).exists(), reason="plant reconstruction missing")


@pytest.fixture(scope="module")
def raw():
    return load_colmap(SCENE, res="2000", model=MODEL, normalize=False, load_images=False)


@pytest.fixture(scope="module")
def norm():
    return load_colmap(SCENE, res="1600", model=MODEL, normalize=True, load_images=True)


def test_split_every_8th(norm):
    n = len(norm["train_cams"]) + len(norm["test_cams"])
    assert n == 216 and len(norm["test_cams"]) == 27
    assert [c.name for c in norm["test_cams"]][:3] == ["frame_000.jpg", "frame_008.jpg", "frame_016.jpg"]


def test_images_masks_and_intrinsics(norm):
    imgs, masks = norm["train_images"], norm["train_masks"]
    assert imgs.shape == (189, 1600, 1200, 3) and imgs.dtype == torch.uint8
    assert masks.shape == (189, 1600, 1200) and masks.dtype == torch.uint8
    c = norm["train_cams"][0]
    assert (c.width, c.height) == (1200, 1600)
    assert abs(c.K[0, 0].item() - 2110.5) < 1.0          # undistorted pinhole focal at 1600 px


def test_projection_matches_colmap_observations(raw):
    """Project COLMAP's 3D points with my world_to_camera + pinhole into the undistorted
    camera, and compare with COLMAP's measured 2D keypoints after undistorting them with
    the same model. The mean error must match COLMAP's reported ~1.0 px."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(SCENE / MODEL))
    (cam,) = rec.cameras.values()
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    K_dist = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    by_name = {c.name: c for c in raw["train_cams"] + raw["test_cams"]}
    rng = np.random.default_rng(0)
    pts = list(rec.points3D.values())
    errs = []
    for idx in rng.choice(len(pts), 400, replace=False):
        p = pts[idx]
        for el in p.track.elements[:3]:
            im = rec.images[el.image_id]
            c = by_name[im.name]
            uv_obs = np.asarray(im.points2D[el.point2D_idx].xy, dtype=np.float64)[None, None]
            uv_und = cv2.undistortPoints(uv_obs, K_dist, np.array([k1, k2, p1, p2]),
                                         P=c.K.double().numpy())[0, 0]
            X = torch.tensor(p.xyz, dtype=torch.float64)[None]
            uv = project_points(world_to_camera(X, c.viewmat.double()), c.K.double())[0].numpy()
            errs.append(np.linalg.norm(uv - uv_und))
    errs = np.array(errs)
    assert np.median(errs) < 1.0 and errs.mean() < 1.5, (np.median(errs), errs.mean())


def test_normalization(norm):
    centers = torch.stack([c.center for c in norm["train_cams"] + norm["test_cams"]])
    assert abs(centers.norm(dim=-1).mean().item() - 4.0) < 1e-3
    ups = torch.stack([-c.viewmat[:3, :3].T @ torch.tensor([0.0, 1.0, 0.0])
                       for c in norm["train_cams"]])                      # camera "up" in world
    assert (torch.nn.functional.normalize(ups.mean(0), dim=0) @ torch.tensor([0.0, 1.0, 0.0])) > 0.99
    for c in norm["train_cams"][:20]:
        R = c.viewmat[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
    # the plant (origin) is in front of every camera, near the image centre
    for c in norm["train_cams"]:
        o = c.viewmat[:3, 3]
        assert o[2] > 0


def test_normalized_projection_is_unchanged(raw, norm):
    """The similarity transform must not change any projection: a raw SfM point and its
    normalized copy land on the same pixel (up to the 2000 -> 1600 rescale, s = 0.8)."""
    T = norm["normalization"]
    pts_raw = torch.as_tensor(raw["points"][:200], dtype=torch.float64)
    pts_n = T["scale"] * (pts_raw - torch.tensor(T["center"], dtype=torch.float64)) \
        @ torch.tensor(T["rotation"], dtype=torch.float64).T
    cr = {c.name: c for c in raw["train_cams"]}
    for c in norm["train_cams"][:10]:
        a = project_points(world_to_camera(pts_raw, cr[c.name].viewmat.double()), cr[c.name].K.double())
        pc = world_to_camera(pts_n, c.viewmat.double())
        b = project_points(pc, c.K.double())
        # Only points actually in view: near or behind the camera plane, u = f x / z blows
        # float32 rounding up to ~1e5 px off-image (the renderer culls those anyway).
        vis = (pc[:, 2] > 0.5) & (b[:, 0] >= 0) & (b[:, 0] < c.width) & (b[:, 1] >= 0) & (b[:, 1] < c.height)
        assert vis.sum() > 50
        # float32 viewmats and per-resolution K_new (0.8 K_2000 ~ K_1600 to ~1e-3 px)
        assert (0.8 * a - b)[vis].abs().max() < 0.01
