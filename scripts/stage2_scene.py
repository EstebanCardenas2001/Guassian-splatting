"""Stage 2 visual check: render a hand-placed scene from several views with render_torch.

Scene (world frame, +y up):
  * coordinate axes built from elongated Gaussians: +x red, +y green, +z blue, white origin
  * a thin yellow disk tilted 45 deg about x (anisotropic: seen face-on, oblique, edge-on)
  * a ring of 12 magenta/cyan Gaussians around the y axis (tests depth ordering/occlusion)

Writes docs/figures/stage2_multiview.png, docs/figures/stage2_turntable.gif and
docs/figures/stage2_footprints.png (projected 3-sigma ellipses over a render).
"""
import math
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.camera import orbit_cameras  # noqa: E402
from src.render_torch import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
DEV = "cuda"


def quat_axis_angle(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return [math.cos(h), *(math.sin(h) * axis)]


def build_scene():
    means, quats, scales, opac, cols = [], [], [], [], []

    def add(m, q, s, o, c):
        means.append(m); quats.append(q); scales.append(s); opac.append(o); cols.append(c)

    ident = [1.0, 0, 0, 0]
    add([0, 0, 0], ident, [0.06] * 3, 0.95, [1, 1, 1])
    for axis, color in ((0, [0.9, 0.1, 0.1]), (1, [0.1, 0.8, 0.2]), (2, [0.15, 0.3, 1.0])):
        for t in np.linspace(0.2, 1.0, 5):
            m = [0.0, 0.0, 0.0]; m[axis] = float(t)
            s = [0.025] * 3; s[axis] = 0.08
            add(m, ident, s, 0.9, color)
    # tilted disk: flat along its local z, rotated 45 deg about world x
    add([0.0, -0.6, 0.0], quat_axis_angle([1, 0, 0], 45), [0.45, 0.45, 0.01], 0.7, [1.0, 0.8, 0.1])
    # ring around the y axis, alternating colors
    for i in range(12):
        a = 2 * math.pi * i / 12
        add([1.4 * math.cos(a), -0.2, 1.4 * math.sin(a)], quat_axis_angle([0, 1, 0], -math.degrees(a)),
            [0.05, 0.12, 0.2], 0.85, [0.9, 0.2, 0.8] if i % 2 == 0 else [0.1, 0.8, 0.9])
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=DEV)
    return t(means), t(quats), t(scales), t(opac), t(cols)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    bg = torch.ones(3, device=DEV)
    W, H = 480, 360

    # Multi-view grid: 6 azimuths at 20 deg elevation + a top view
    cams = orbit_cameras(6, 4.5, 20.0, 50.0, W, H, device=DEV)
    cams += orbit_cameras(1, 4.5, 85.0, 50.0, W, H, device=DEV)
    titles = [f"azimuth {60 * i}°" for i in range(6)] + ["top (elev. 85°)"]
    imgs = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for cam in cams:
        img, _, info = render(*scene, cam.viewmat, cam.K, W, H, bg=bg)
        imgs.append(img.clamp(0, 1).cpu().numpy())
    torch.cuda.synchronize()
    ms = 1e3 * (time.perf_counter() - t0) / len(cams)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), dpi=110)
    for ax, im, title in zip(axes.flat, imgs, titles):
        ax.imshow(im); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.02, 0.5, "+x red, +y green, +z blue\n"
                       "yellow: thin disk tilted 45° about x\n"
                       "ring: 12 anisotropic Gaussians\n\n"
                       f"{len(scene[0])} Gaussians, {W}×{H}\n{ms:.1f} ms / view (pure PyTorch)",
                       fontsize=10, va="center")
    fig.tight_layout()
    fig.savefig(OUT / "stage2_multiview.png")

    # Turntable GIF
    frames = []
    for cam in orbit_cameras(48, 4.5, 20.0, 50.0, 320, 240, device=DEV):
        img, _, _ = render(*scene, cam.viewmat, cam.K, 320, 240, bg=bg)
        frames.append((img.clamp(0, 1) * 255).byte().cpu().numpy())
    iio.imwrite(OUT / "stage2_turntable.gif", np.stack(frames), duration=60, loop=0)

    # Footprints: projected 3-sigma ellipses (from cov2d) over the render of view 1
    cam = cams[1]
    img, _, info = render(*scene, cam.viewmat, cam.K, W, H, bg=bg)
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=110)
    ax.imshow(img.clamp(0, 1).cpu().numpy())
    t = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(t), np.sin(t)])
    for m, cov, ok in zip(info["means2d"].cpu().numpy(), info["cov2d"].cpu().numpy(),
                          info["valid"].cpu().numpy()):
        if not ok:
            continue
        L = np.linalg.cholesky(cov)
        e = m[:, None] + 3 * L @ circle
        ax.plot(e[0], e[1], lw=0.8, color="#222222")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("EWA footprints: projected 3σ ellipses (azimuth 60°)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "stage2_footprints.png")
    print(f"{len(scene[0])} Gaussians, {ms:.1f} ms/view at {W}x{H}; figures -> {OUT}")


if __name__ == "__main__":
    main()
