"""Deliverables: .ply in the standard 3DGS layout, and a fly-through video along interpolated
capture poses (Stage 9).

    python -m src.export --run outputs/<run> --ply plant.ply --video plant_orbit.mp4

PLY layout (binary little-endian, one float32 per property, as written by the original 3DGS
code and read by SuperSplat and most viewers):
    x y z | nx ny nz (zeros) | f_dc_0..2 | f_rest_0..(3K-1) | opacity | scale_0..2 | rot_0..3
* f_dc: the degree-0 SH coefficient per channel (color = C0 * f_dc + 1/2);
* f_rest: the higher-order coefficients **channel-major**: f_rest_{c*K + k} is coefficient k
  of channel c, i.e. shN (N, K, 3) -> permute to (N, 3, K) -> flatten;
* opacity and scales are stored *raw*: logit(opacity) and log(scale); rot is the
  (unnormalized) quaternion w, x, y, z.
Positions are in the normalized scene frame (plant at the origin, +y up, cameras at
radius ~4), so the model opens upright in y-up viewers.

Fly-through: keyframes are capture cameras in the middle elevation band, ordered by azimuth
around +y. Positions follow a periodic cubic spline through them and orientations a C2
rotation spline (scipy RotationSpline) through their camera-to-world rotations, so the
video retraces how the photos were taken, smoothed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.gaussians import Gaussians

PROPS_FIXED = ["x", "y", "z", "nx", "ny", "nz"]


def ply_property_names(K: int) -> list[str]:
    """Property order of the 3DGS PLY for K higher-order SH coefficients per channel."""
    return (PROPS_FIXED + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(3 * K)]
            + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])


def write_ply(model: Gaussians, path, min_opacity: float = 0.0) -> int:
    """Write ``model`` (SH colors) as a 3DGS PLY. Gaussians with opacity < ``min_opacity``
    (invisible: below 1/255 they never contribute) can be dropped. Returns the count."""
    if not model.uses_sh:
        raise ValueError("3DGS PLYs store SH coefficients; train with sh_degree set")
    p = {k: v.detach().float().cpu() for k, v in model.params.items()}
    keep = torch.sigmoid(p["opacities"]) >= min_opacity
    N, K = int(keep.sum()), p["shN"].shape[1]
    cols = [p["means"][keep], torch.zeros(N, 3), p["sh0"][keep].reshape(N, 3),
            p["shN"][keep].permute(0, 2, 1).reshape(N, 3 * K), p["opacities"][keep][:, None],
            p["scales"][keep], p["quats"][keep]]
    data = torch.cat(cols, 1).numpy().astype("<f4")
    names = ply_property_names(K)
    assert data.shape[1] == len(names)
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {N}\n"
              + "".join(f"property float {n}\n" for n in names) + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())
    return N


def read_ply(path) -> Gaussians:
    """Read a 3DGS PLY (any property order) back into a Gaussians model."""
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    col = lambda n: torch.from_numpy(np.asarray(v[n], dtype=np.float32))
    names = [p.name for p in v.properties]
    K = sum(n.startswith("f_rest_") for n in names) // 3
    stack = lambda prefix, n: torch.stack([col(f"{prefix}{i}") for i in range(n)], 1)
    shN = stack("f_rest_", 3 * K).reshape(-1, 3, K).permute(0, 2, 1).contiguous()
    return Gaussians(torch.stack([col("x"), col("y"), col("z")], 1),
                     stack("rot_", 4), stack("scale_", 3), col("opacity"),
                     sh0=stack("f_dc_", 3)[:, None, :], shN=shN)


# ----------------------------------------------------------------------------- fly-through
def keyframes(cams, elev_band=(10.0, 45.0), n_key: int = 16):
    """Capture cameras with elevation in ``elev_band`` (degrees above the plant's horizon,
    +y up), one per azimuth sector, ordered by azimuth. Returns (positions, c2w rotations)."""
    C = np.stack([c.center.cpu().numpy() for c in cams])
    R = np.stack([c.viewmat[:3, :3].cpu().numpy().T for c in cams])          # camera -> world
    r = np.linalg.norm(C, axis=1)
    elev = np.degrees(np.arcsin(C[:, 1] / r))
    az = np.arctan2(C[:, 0], C[:, 2])
    idx = np.where((elev >= elev_band[0]) & (elev <= elev_band[1]))[0]
    sectors = np.floor((az[idx] + np.pi) / (2 * np.pi) * n_key).astype(int)
    keys = []
    for s in range(n_key):                                 # the camera nearest the band middle
        cand = idx[sectors == s]
        if len(cand):
            keys.append(cand[np.argmin(np.abs(elev[cand] - np.mean(elev_band)))])
    keys = sorted(keys, key=lambda i: az[i])
    return C[keys], R[keys], az[keys]


def interpolate_path(C, R, az, n_frames: int, rot_weight: float = 2.0):
    """Closed, smooth camera path through the keyframes: periodic cubic spline for positions
    over the unwrapped azimuth, C2 rotation spline for orientations (keyframes wrapped on both
    sides so the loop closes without a kink), sampled at constant speed. Returns lists of
    (position, c2w rotation)."""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, RotationSpline
    t = np.unwrap(az)
    t_loop = np.append(t, t[0] + 2 * np.pi)
    pos = CubicSpline(t_loop, np.vstack([C, C[:1]]), bc_type="periodic")
    k = len(t)
    t_w = np.concatenate([t[-3:] - 2 * np.pi, t, t[:3] + 2 * np.pi])
    rots = Rotation.from_matrix(np.concatenate([R[-3:], R, R[:3]]))
    rot = RotationSpline(t_w, rots)
    # Constant-speed reparameterization: sample densely, measure the path length in combined
    # units (scene distance + rot_weight * radians), and pick frames at equal length steps, so
    # the camera doesn't rush past close keyframes and crawl past far ones.
    dense = np.linspace(t[0], t[0] + 2 * np.pi, 20 * n_frames + 1)
    P = pos(dense)
    ang = (rot(dense[1:]) * rot(dense[:-1]).inv()).magnitude()
    step = np.linalg.norm(np.diff(P, axis=0), axis=1) + rot_weight * ang
    s = np.concatenate([[0.0], np.cumsum(step)])
    ts = np.interp(np.linspace(0, s[-1], n_frames, endpoint=False), s, dense)
    return [(pos(x), rot(x).as_matrix()) for x in ts]


@torch.no_grad()
def render_path(model: Gaussians, path, K: torch.Tensor, W: int, H: int, bg=(1.0, 1.0, 1.0)):
    from src.render_gsplat import render
    geo = [t.detach() for t in model.geometry()]
    bgt = torch.tensor(bg, device="cuda")
    frames = []
    for p, Rc2w in path:
        V = torch.eye(4, device="cuda")
        V[:3, :3] = torch.as_tensor(Rc2w.T, dtype=torch.float32)
        V[:3, 3] = torch.as_tensor(-Rc2w.T @ p, dtype=torch.float32)
        img, _, _ = render(*geo, model.colors_for(V), V, K, W, H, bg=bgt)
        frames.append((img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ply")
    ap.add_argument("--video")
    ap.add_argument("--gif", help="optional small GIF of the same path")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--min_opacity", type=float, default=1 / 255)
    args = ap.parse_args()
    import imageio.v3 as iio
    from omegaconf import OmegaConf

    from src.camera import load_colmap
    from src.run import REPO_ROOT
    run = Path(args.run)
    model = Gaussians.from_state_dict(torch.load(run / "ckpt.pt", map_location="cuda", weights_only=False)["model"]).cuda()
    if args.ply:
        n = write_ply(model, args.ply, args.min_opacity)
        print(f"{n} of {len(model)} Gaussians -> {args.ply} ({Path(args.ply).stat().st_size / 2**20:.1f} MB)")
    if args.video or args.gif:
        cfg = OmegaConf.load(run / "config.yaml")
        d = load_colmap(REPO_ROOT / cfg.data.path, res="1600", model=cfg.data.model, load_images=False)
        cams = d["train_cams"] + d["test_cams"]
        C, R, az = keyframes(cams)
        H = args.height
        W = int(round(H * cams[0].width / cams[0].height)) // 2 * 2
        s = H / cams[0].height
        K = torch.tensor([[cams[0].K[0, 0] * s, 0, W / 2], [0, cams[0].K[1, 1] * s, H / 2], [0, 0, 1]],
                         device="cuda", dtype=torch.float32)
        frames = render_path(model, interpolate_path(C, R, az, args.frames), K, W, H)
        if args.video:
            iio.imwrite(args.video, frames, fps=30, codec="libx264", quality=8)
            print(f"{len(frames)} frames {W}x{H} along {len(C)} keyframes -> {args.video}")
        if args.gif:
            iio.imwrite(args.gif, frames[::5, ::4, ::4], duration=100, loop=0)
            print(f"gif -> {args.gif}")


if __name__ == "__main__":
    main()
