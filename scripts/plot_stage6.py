"""Stage 6 figures: camera rig + sparse cloud, SfM points reprojected onto photos, masks.

    python -m scripts.plot_stage6
Writes docs/figures/stage6_*.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.camera import load_colmap  # noqa: E402
from src.render_torch import project_points, world_to_camera  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
INK, MUTED = "#0b0b0b", "#52514e"
TRAIN, TEST = "#2a78d6", "#eb6834"


def frustum(cam, depth=0.6):
    """World-space frustum lines of a camera (apex + 4 image corners at ``depth``)."""
    R, t = cam.viewmat[:3, :3].numpy(), cam.viewmat[:3, 3].numpy()
    K = cam.K.numpy()
    corners = np.array([[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]], float)
    rays = np.c_[(corners[:, 0] - K[0, 2]) / K[0, 0], (corners[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)]
    C = -R.T @ t
    pts = C + (rays * depth) @ R                      # camera -> world: R^T x
    segs = [(C, p) for p in pts] + [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return C, segs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_colmap(ROOT / "data/plant", res="1600", load_images=True)
    P, col = d["points"], d["point_colors"] / 255.0
    keep = np.linalg.norm(P, axis=1) < 3.0                     # drop far background points for the plot
    rng = np.random.default_rng(0)
    sel = rng.choice(np.where(keep)[0], min(40000, keep.sum()), replace=False)

    # ---- 1. camera rig + sparse cloud (normalized frame: plant at origin, +y up)
    fig = plt.figure(figsize=(13, 6), dpi=150)
    for k, (elev, azim, title) in enumerate(((20, -60, "oblique"), (89, -90, "top view"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(P[sel, 0], P[sel, 2], P[sel, 1], c=col[sel], s=0.3, depthshade=False)
        for cams, c in ((d["train_cams"], TRAIN), (d["test_cams"], TEST)):
            for cam in cams:
                C, segs = frustum(cam)
                for a, b in segs:
                    ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], color=c, lw=0.5)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.set_zlim(-2, 5)
        ax.set_box_aspect((1, 1, 0.78)); ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)")
        ax.set_title(f"{title}: {len(d['train_cams'])} train (blue), {len(d['test_cams'])} test (orange) cameras",
                     fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "stage6_cameras.png"); plt.close(fig)

    # ---- 2. points COLMAP observed in each photo, projected with my model (full + zoom)
    import pycolmap
    rec = pycolmap.Reconstruction(str(ROOT / "data/plant/colmap_incremental/sparse/1"))
    T = d["normalization"]
    c0, Rn, sc = np.array(T["center"]), np.array(T["rotation"]), T["scale"]
    by_name = {im.name: im for im in rec.images.values()}
    fig, axes = plt.subplots(2, 3, figsize=(13, 11), dpi=150)
    for j, i in enumerate((3, 60, 150)):
        cam, img = d["train_cams"][i], d["train_images"][i].numpy()
        im = by_name[cam.name]
        xyz = np.array([rec.points3D[p.point3D_id].xyz for p in im.points2D if p.has_point3D()])
        xyz = sc * (xyz - c0) @ Rn.T                          # into the normalized frame
        uv = project_points(world_to_camera(torch.as_tensor(xyz, dtype=torch.float32), cam.viewmat),
                            cam.K).numpy()
        cx, cy = np.median(uv, 0).astype(int)                 # zoom on the densest region
        h = 140
        for r, (ax, crop) in enumerate(((axes[0, j], None), (axes[1, j], (cx - h, cx + h, cy - h, cy + h)))):
            ax.imshow(img)
            ax.scatter(uv[:, 0], uv[:, 1], s=1.5 if crop is None else 10, c="#00e5ff",
                       marker="." if crop is None else "+", linewidths=0.8)
            if crop is not None:
                ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[3], crop[2])
            else:
                ax.add_patch(plt.Rectangle((cx - h, cy - h), 2 * h, 2 * h, fill=False, ec="#eb6834", lw=1.2))
                ax.set_title(f"{cam.name}: {len(uv)} observed points", fontsize=9, color=MUTED)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Points COLMAP matched in each photo, re-projected with my camera model (cyan); "
                 "bottom: 280×280 px zoom", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "stage6_reprojection.png"); plt.close(fig)

    # ---- 3. masks: 12 frames spread over the capture, background tinted
    idx = np.linspace(0, len(d["train_cams"]) - 1, 12).round().astype(int)
    fig, axes = plt.subplots(2, 6, figsize=(15, 7.6), dpi=130, gridspec_kw={"hspace": 0.25})
    for ax, i in zip(axes.flat, idx):
        img = d["train_images"][i].numpy().astype(np.float32) / 255
        m = d["train_masks"][i].numpy().astype(np.float32)[..., None] / 255
        over = img * m + (1 - m) * (0.35 * img + 0.65 * np.array([1.0, 0.0, 1.0]))
        ax.imshow(over); ax.set_title(f"{d['train_cams'][i].name}  fg {m.mean():.2f}", fontsize=8, color=MUTED)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("BiRefNet masks (background tinted magenta), undistorted 1600 px", fontsize=10, color=INK)
    fig.savefig(OUT / "stage6_masks.png", bbox_inches="tight"); plt.close(fig)
    print(f"{len(P)} points ({keep.sum()} within r<3), figures -> {OUT}")


if __name__ == "__main__":
    main()
