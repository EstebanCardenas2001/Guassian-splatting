"""Stage 8: memory/time of one full training step vs resolution and Gaussian count.

    python -m scripts.profile_stage8 --ckpt outputs/<stage7 D run>/ckpt.pt --out file.json
Gaussians are resampled (with replacement, small positional jitter) from a trained plant
model with SH degree 3, so footprints and depth complexity are realistic. One step =
render (gsplat, packed) + L1/SSIM loss on a real photo + backward + Adam step on all
parameters (moments included in the memory). OOM is caught and reported.
"""
import argparse
import json
import time

import torch

from src.camera import load_colmap
from src.gaussians import Gaussians
from src.losses import photometric_loss
from src.render_gsplat import render
from src.run import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--res", default="1600,2000,full")
    ap.add_argument("--ns", default="500000,1000000,2000000,3000000,4000000,6000000")
    ap.add_argument("--antialiased", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    base = Gaussians.from_state_dict(torch.load(args.ckpt, map_location="cuda", weights_only=False)["model"])
    gen = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for res in args.res.split(","):
        d = load_colmap(REPO_ROOT / "data/plant", res=res, load_split="test")
        cam = d["test_cams"][1].to("cuda")
        target = d["test_images"][1].cuda().float() / 255
        bg = torch.ones(3, device="cuda")
        for n in map(int, args.ns.split(",")):
            row = {"res": f"{cam.width}x{cam.height}", "n": n, "antialiased": args.antialiased}
            model = opts = None
            try:
                idx = torch.randint(0, len(base), (n,), device="cuda", generator=gen)
                p = {k: base.params[k].detach()[idx].clone() for k in base.params}
                p["means"] += 0.002 * torch.randn(p["means"].shape, device="cuda", generator=gen)
                model = Gaussians(p["means"], p["quats"], p["scales"], p["opacities"], sh0=p["sh0"], shN=p["shN"]).cuda()
                opts = [torch.optim.Adam([q], lr=1e-4, eps=1e-15) for q in model.params.values()]
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                times = []
                for i in range(6):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    img, _, _ = render(*model.geometry(), model.colors_for(cam.viewmat), cam.viewmat, cam.K,
                                       cam.width, cam.height, bg=bg, antialiased=args.antialiased)
                    photometric_loss(img, target).backward()
                    for o in opts:
                        o.step(); o.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    if i > 0:
                        times.append(time.perf_counter() - t0)
                row.update(ms=1e3 * sum(times) / len(times), peak_GB=torch.cuda.max_memory_allocated() / 2**30,
                           status="ok")
            except torch.OutOfMemoryError:
                row.update(status="OOM")
            del model, opts
            torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            rows.append(row)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
