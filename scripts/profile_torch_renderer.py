"""Profile render_torch forward+backward (L1 + D-SSIM) on lego cameras: ms/iter, peak VRAM,
(Gaussian, pixel) pairs, for a grid of Gaussian counts and resolutions. OOM is caught.

    python -m scripts.profile_torch_renderer [--ckpt outputs/<run>/ckpt.pt] [--out file.json]

Without --ckpt, Gaussians are 3DGS-style random init: uniform in [-1.3, 1.3]^3 with
isotropic scale = mean spacing (2.6 / N^(1/3)), opacity 0.1. With --ckpt, the trained
model's Gaussians are randomly subsampled / repeated to each N (a "trained-like" state).
"""
import argparse
import json
import time

import torch

from src.camera import load_blender
from src.gaussians import Gaussians
from src.losses import photometric_loss
from src.render_torch import render as render_torch
from src.run import REPO_ROOT


def make_gaussians(n, ckpt, gen, scale_factor=1.0):
    if ckpt is None:
        lo, hi = torch.full((3,), -1.3, device="cuda"), torch.full((3,), 1.3, device="cuda")
        return Gaussians.random_box(n, lo, hi, scale=scale_factor * 2.6 / n ** (1 / 3),
                                    opacity=0.1, generator=gen)
    sd = torch.load(ckpt, map_location="cuda")["model"]
    idx = torch.randint(0, sd["means"].shape[0], (n,), device="cuda", generator=gen)
    return Gaussians(*(sd[k][idx].clone() for k in
                       ("means", "quats", "log_scales", "opacity_logits", "color_logits")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--ns", default="10000,30000,100000,300000,1000000")
    ap.add_argument("--downscales", default="4,2,1")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--scale_factors", default="1.0", help="init scale as a multiple of mean spacing")
    ap.add_argument("--max_pairs", type=int, default=80_000_000)
    ap.add_argument("--backend", default="torch", choices=["torch", "gsplat"])
    ap.add_argument("--out")
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    if args.backend == "gsplat":
        from src.render_gsplat import render as render_gs
        render = lambda *a, max_pairs=None, **k: render_gs(*a, **k)
    else:
        render = render_torch
    rows = []
    for ds in map(int, args.downscales.split(",")):
        cams, imgs = load_blender(REPO_ROOT / "data/nerf_synthetic/lego", "train", downscale=ds)
        target = imgs[0].cuda().float() / 255
        cam = cams[0].to("cuda")
        bg = torch.ones(3, device="cuda")
        for n, sf in [(n, sf) for sf in map(float, args.scale_factors.split(","))
                      for n in map(int, args.ns.split(","))]:
            row = {"res": cam.width, "n": n, "scale_factor": sf, "backend": args.backend}
            model = None
            try:
                model = make_gaussians(n, args.ckpt, gen, sf)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                times = []
                for i in range(args.iters + 1):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    img, _, info = render(*model.activated(), cam.viewmat, cam.K, cam.width, cam.height, bg=bg,
                                          max_pairs=args.max_pairs)
                    photometric_loss(img, target).backward()
                    torch.cuda.synchronize()
                    if i > 0:
                        times.append(time.perf_counter() - t0)
                    model.zero_grad(set_to_none=True)
                row.update(ms=1e3 * sum(times) / len(times), pairs_M=info["pairs"] / 1e6,
                           peak_GB=(torch.cuda.max_memory_allocated() - base) / 2**30, status="ok")
            except torch.OutOfMemoryError:
                row.update(status="OOM", peak_GB=torch.cuda.max_memory_allocated() / 2**30)
            except RuntimeError as e:
                row.update(status=f"error: {str(e)[:80]}")
            del model
            torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            rows.append(row)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
