"""Train 3D Gaussians on a posed image set.

    python -m src.train --config configs/stage4_lego_torch.yaml [key=value ...]
    python -m src.train --resume outputs/<run_dir>          # continue an interrupted run

One random training view per iteration (each epoch visits every view once, in random
order), loss (1 - lambda) L1 + lambda (1 - SSIM), per-parameter Adam learning rates with
exponential decay on the position LR. Training images live on the GPU as uint8.
A checkpoint (model, optimizer, iteration, RNG state) is written every ``ckpt_every``
iterations so the VM can be interrupted.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.camera import load_blender
from src.config import load_config
from src.gaussians import Gaussians
from src.losses import photometric_loss, psnr, ssim
from src.run import REPO_ROOT, make_run_dir, seed_everything, write_metrics

PARAMS = ("means", "quats", "log_scales", "opacity_logits", "color_logits")


def get_renderer(cfg):
    if cfg.render.backend == "torch":
        from src.render_torch import render
    elif cfg.render.backend == "gsplat":
        from src.render_gsplat import render
    else:
        raise ValueError(cfg.render.backend)
    eps2d, aa = cfg.render.eps2d, cfg.render.antialiased
    return lambda model, cam, bg: render(*model.activated(), cam.viewmat, cam.K, cam.width,
                                         cam.height, bg=bg, eps2d=eps2d, antialiased=aa)


def load_data(cfg, device):
    d = cfg.data
    if d.type != "blender":
        raise ValueError(d.type)
    path = REPO_ROOT / d.path
    train_cams, train_imgs = load_blender(path, "train", d.downscale, tuple(d.bg))
    test_cams, test_imgs = load_blender(path, "test", d.downscale, tuple(d.bg))
    return ([c.to(device) for c in train_cams], train_imgs.to(device),
            [c.to(device) for c in test_cams], test_imgs.to(device))


def init_model(cfg, device, gen) -> Gaussians:
    m = cfg.model
    if m.init != "random_box":
        raise ValueError(m.init)
    lo = torch.full((3,), float(m.box[0]), device=device)
    hi = torch.full((3,), float(m.box[1]), device=device)
    spacing = float(torch.prod(hi - lo)) ** (1 / 3) / m.n ** (1 / 3)
    return Gaussians.random_box(m.n, lo, hi, scale=m.scale_factor * spacing, opacity=m.opacity,
                                generator=gen)


@torch.no_grad()
def evaluate(render_fn, model, cams, imgs, bg, idx, out_dir: Path | None = None, save_first: int = 0):
    ps, ss = [], []
    for j, i in enumerate(idx):
        img, _, _ = render_fn(model, cams[i], bg)
        ref = imgs[i].float() / 255
        ps.append(psnr(img, ref).item())
        ss.append(ssim(img.clamp(0, 1), ref).item())
        if out_dir is not None and j < save_first:
            pair = torch.cat([ref, img.clamp(0, 1)], 1)
            iio.imwrite(out_dir / f"test_{i:03d}.png", (pair * 255 + 0.5).byte().cpu().numpy())
    return float(np.mean(ps)), float(np.mean(ss))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage4_lego_torch.yaml")
    ap.add_argument("--resume", help="run directory to resume from its ckpt.pt")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.resume)
        cfg = OmegaConf.load(run_dir / "config.yaml")
        if args.overrides:          # e.g. train.iterations=30000 to extend a finished run
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
            OmegaConf.save(cfg, run_dir / "config.yaml")
    else:
        cfg = load_config(REPO_ROOT / args.config, args.overrides)
        run_dir = make_run_dir(cfg)
    seed_everything(cfg.seed)
    dev = torch.device(cfg.device)
    tb = SummaryWriter(run_dir / "tb")
    tc = cfg.train

    train_cams, train_imgs, test_cams, test_imgs = load_data(cfg, dev)
    bg = torch.tensor(list(cfg.data.bg), device=dev)
    render_fn = get_renderer(cfg)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    model = init_model(cfg, dev, gen)
    opt = torch.optim.Adam([{"params": [getattr(model, k)], "lr": tc.lr[k], "name": k} for k in PARAMS],
                           eps=1e-15)
    means_group = next(g for g in opt.param_groups if g["name"] == "means")
    gamma = (tc.means_lr_final / tc.lr.means) ** (1.0 / tc.iterations)
    eval_idx = np.linspace(0, len(test_cams) - 1, cfg.data.eval_subset).round().astype(int).tolist()
    view_gen = torch.Generator().manual_seed(cfg.seed)
    start, order, elapsed = 0, [], 0.0

    ckpt_path = run_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start, order, elapsed = ck["iteration"], ck["order"], ck["elapsed"]
        view_gen.set_state(ck["view_gen"].cpu())       # CPU generator; map_location moved it
        print(f"resumed at iteration {start}")

    def save_ckpt(it):
        tmp = run_dir / "ckpt.pt.tmp"
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "iteration": it,
                    "order": order, "elapsed": elapsed, "view_gen": view_gen.get_state()}, tmp)
        tmp.replace(ckpt_path)          # atomic: an interruption never leaves a torn checkpoint

    print(f"{len(model)} Gaussians, {len(train_cams)} train / {len(test_cams)} test views at "
          f"{train_cams[0].width}x{train_cams[0].height}, backend={cfg.render.backend}")
    torch.cuda.reset_peak_memory_stats()
    t_last = time.perf_counter()
    for it in range(start + 1, tc.iterations + 1):
        if not order:
            order = torch.randperm(len(train_cams), generator=view_gen).tolist()
        i = order.pop()
        img, alpha, info = render_fn(model, train_cams[i], bg)
        ref = train_imgs[i].float() / 255
        loss = photometric_loss(img, ref, tc.lambda_dssim)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        means_group["lr"] = tc.lr.means * gamma ** it

        if it % tc.log_every == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            dt = (now - t_last) / tc.log_every
            elapsed += now - t_last
            t_last = now
            n_vis = int((info["radii"] > 0).sum())
            tb.add_scalar("train/loss", loss.item(), it)
            tb.add_scalar("train/psnr", psnr(img.detach(), ref).item(), it)
            tb.add_scalar("perf/ms_per_iter", 1e3 * dt, it)
            tb.add_scalar("perf/pairs_M", info["pairs"] / 1e6, it)
            tb.add_scalar("perf/peak_mem_GB", torch.cuda.max_memory_allocated() / 2**30, it)
            tb.add_scalar("model/visible_gaussians", n_vis, it)
            tb.add_scalar("model/opacity_gt_1_255",
                          int((torch.sigmoid(model.opacity_logits) > 1 / 255).sum()), it)
            print(f"[{it:6d}] loss {loss.item():.4f}  {1e3 * dt:6.0f} ms/it  pairs {info['pairs'] / 1e6:5.1f}M  "
                  f"visible {n_vis:7d}  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GB", flush=True)
        if it % tc.eval_every == 0 or it == tc.iterations:
            p, s = evaluate(render_fn, model, test_cams, test_imgs, bg, eval_idx)
            tb.add_scalar("test/psnr", p, it)
            tb.add_scalar("test/ssim", s, it)
            print(f"[{it:6d}] test PSNR {p:.2f}  SSIM {s:.4f} (subset of {len(eval_idx)})", flush=True)
            t_last = time.perf_counter()        # don't bill evaluation to training time
        if it % tc.ckpt_every == 0 or it == tc.iterations:
            save_ckpt(it)
            t_last = time.perf_counter()

    p, s = evaluate(render_fn, model, test_cams, test_imgs, bg, range(len(test_cams)),
                    out_dir=run_dir / "renders", save_first=8)
    o = torch.sigmoid(model.opacity_logits)
    metrics = {"test_psnr": p, "test_ssim": s, "n_test_views": len(test_cams),
               "resolution": [train_cams[0].width, train_cams[0].height],
               "iterations": tc.iterations, "train_time_s": elapsed,
               "ms_per_iter_mean": 1e3 * elapsed / max(tc.iterations - start, 1) if start == 0 else None,
               "peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30,
               "n_gaussians": len(model), "n_opacity_gt_1_255": int((o > 1 / 255).sum()),
               "backend": cfg.render.backend}
    write_metrics(run_dir, metrics)
    tb.close()
    print(json.dumps(metrics, indent=1))
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
