"""Turntable of a trained model: MP4 + GIF from cameras orbiting the origin (+y up).

    python -m scripts.render_orbit --run outputs/<run> --out docs/figures/plant_orbit
The scene is normalized (plant at the origin, up = +y), so a circle of radius r at
elevation e around the origin frames it like the capture. Field of view: the training
camera's vertical FOV.
"""
import argparse
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from src.camera import focal_from_fov, orbit_cameras
from src.gaussians import Gaussians
from src.render_gsplat import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--radius", type=float, default=3.4)
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--vfov", type=float, default=41.5, help="vertical FOV in degrees (training: 41.5)")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--target_y", type=float, default=0.15, help="look slightly above the origin")
    args = ap.parse_args()
    ck = torch.load(Path(args.run) / "ckpt.pt", map_location="cuda", weights_only=False)
    model = Gaussians.from_state_dict(ck["model"]).cuda()
    geo = [t.detach() for t in model.geometry()]
    W, H = args.width, args.height
    hfov = 2 * math.degrees(math.atan(math.tan(math.radians(args.vfov) / 2) * W / H))
    target = torch.tensor([0.0, args.target_y, 0.0], device="cuda")
    cams = orbit_cameras(args.frames, args.radius, args.elevation, hfov, W, H, target=target, device="cuda")
    bg = torch.ones(3, device="cuda")
    frames = []
    with torch.no_grad():
        for c in cams:
            img, _, _ = render(*geo, model.colors_for(c.viewmat), c.viewmat, c.K, W, H, bg=bg)
            frames.append((img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())
    frames = np.stack(frames)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out.with_suffix(".mp4"), frames, fps=30, codec="libx264", quality=8)
    small = frames[::2, ::3, ::3]                                   # GIF: 60 frames at 1/3 size
    iio.imwrite(out.with_suffix(".gif"), small, duration=66, loop=0)
    print(f"{len(frames)} frames {W}x{H} -> {out.with_suffix('.mp4')}, {out.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
