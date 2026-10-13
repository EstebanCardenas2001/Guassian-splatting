"""Train 3D Gaussians on a posed image set (NeRF-synthetic or a COLMAP capture).

    python -m src.train --config configs/stage7_A.yaml [key=value ...]
    python -m src.train --resume outputs/<run_dir>          # continue an interrupted run

One random training view per iteration (each epoch visits every view once, in random
order). Loss:
    L = (1 - lambda) L1(I, I*) + lambda (1 - SSIM(I, I*)) + lambda_mask |alpha - m|_1
where, for masked captures, the target I* = m * photo + (1 - m) * bg is the photo
composited onto the fixed background with the foreground mask m in [0, 1], and the render
uses the same background. The mask term (Stage 7B) asks the rendered opacity to be 1 on
the object and 0 elsewhere; without it, Gaussians the color of the background are free.
Per-parameter Adam learning rates, exponential decay on the position LR, optional
densification (src/densify.py) and progressive spherical harmonics (src/sh.py).
Images and masks live on the GPU as uint8; checkpoints are atomic and resumable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.camera import load_blender, load_colmap
from src.config import load_config
from src.gaussians import Gaussians
from src.losses import photometric_loss, psnr, ssim
from src.run import REPO_ROOT, make_run_dir, seed_everything, write_metrics

# config learning-rate names -> parameter keys (gsplat's names)
LR_KEYS = {"means": "means", "quats": "quats", "log_scales": "scales",
           "opacity_logits": "opacities", "color_logits": "colors", "sh0": "sh0", "shN": "shN"}


# ----------------------------------------------------------------------------- setup
def make_strategy(cfg, train_cams):
    """Densification strategy (None, gsplat's DefaultStrategy, or my ADC) and its state.
    ``scene_scale`` follows 3DGS: 1.1 x the largest camera distance from their centroid."""
    d = cfg.get("densify", {"strategy": "none"})
    if d["strategy"] == "none":
        return None, None
    if cfg.render.backend != "gsplat":
        raise ValueError("densification needs the gsplat backend's meta (screen-space means)")
    kw = {k: v for k, v in d.items() if k not in ("strategy", "scene_scale")}
    if d["strategy"] in ("adc", "mcmc"):
        # configs inherit densify options from their base; keep only this strategy's fields
        import dataclasses
        from src.densify import ADC, MCMC
        fields = {f.name for f in dataclasses.fields(ADC if d["strategy"] == "adc" else MCMC)}
        dropped = sorted(set(kw) - fields)
        if dropped:
            print(f"densify: ignoring options not used by {d['strategy']}: {dropped}")
        kw = {k: v for k, v in kw.items() if k in fields}
    if d["strategy"] == "gsplat_default":
        from gsplat.strategy import DefaultStrategy
        strategy = DefaultStrategy(**kw)
    elif d["strategy"] == "adc":                    # my 3DGS ADC (absgrad=true: AbsGS)
        from src.densify import ADC
        strategy = ADC(**kw)
    elif d["strategy"] == "mcmc":                   # my 3DGS-MCMC, hard cap on N
        from src.densify import MCMC
        strategy = MCMC(**kw)
    else:
        raise ValueError(d["strategy"])
    centers = torch.stack([c.center for c in train_cams])
    scene_scale = d.get("scene_scale") or float(1.1 * (centers - centers.mean(0)).norm(dim=-1).max())
    return strategy, strategy.initialize_state(scene_scale=scene_scale)


def get_renderer(cfg):
    if cfg.render.backend == "torch":
        from src.render_torch import render
    elif cfg.render.backend == "gsplat":
        from src.render_gsplat import render
    else:
        raise ValueError(cfg.render.backend)
    eps2d, aa = cfg.render.eps2d, cfg.render.antialiased
    extra = {"absgrad": True} if cfg.render.get("absgrad", False) else {}   # AbsGS statistic

    def render_fn(model, cam, bg, sh_degree=None):
        colors = model.colors_for(cam.viewmat, sh_degree)
        return render(*model.geometry(), colors, cam.viewmat, cam.K, cam.width, cam.height,
                      bg=bg, eps2d=eps2d, antialiased=aa, **extra)
    return render_fn


def load_data(cfg, device) -> dict:
    """Uniform dict: {train,test}_{cams,imgs,masks} (masks None without masks), points, colors."""
    d = cfg.data
    if d.type == "blender":
        path = REPO_ROOT / d.path
        trc, tri = load_blender(path, "train", d.downscale, tuple(d.bg))
        tec, tei = load_blender(path, "test", d.downscale, tuple(d.bg))
        out = {"train_cams": trc, "train_imgs": tri, "train_masks": None,
               "test_cams": tec, "test_imgs": tei, "test_masks": None, "points": None, "colors": None}
    elif d.type == "colmap":
        c = load_colmap(REPO_ROOT / d.path, res=str(d.res), model=d.model, test_every=d.test_every)
        out = {"train_cams": c["train_cams"], "train_imgs": c["train_images"],
               "test_cams": c["test_cams"], "test_imgs": c["test_images"],
               "train_masks": c["train_masks"] if d.use_masks else None,
               "test_masks": c["test_masks"] if d.use_masks else None,
               "points": torch.as_tensor(c["points"]), "colors": torch.as_tensor(c["point_colors"])}
    else:
        raise ValueError(d.type)
    # Evaluation set: the test split, optionally at another resolution (e.g. full res, so
    # runs trained at different resolutions are compared on the same images).
    eres = str(cfg.get("eval", {}).get("res", None) or d.get("res", ""))
    if d.type == "colmap" and eres != str(d.res):
        e = load_colmap(REPO_ROOT / d.path, res=eres, model=d.model, test_every=d.test_every,
                        load_split="test")
        out.update(eval_cams=e["test_cams"], eval_imgs=e["test_images"],
                   eval_masks=e["test_masks"] if d.use_masks else None)
    else:
        out.update(eval_cams=out["test_cams"], eval_imgs=out["test_imgs"], eval_masks=out["test_masks"])
    # Images live on the GPU as uint8 if they fit; otherwise in pinned CPU memory, copied
    # asynchronously per step (data.cache = gpu | cpu).
    cache = d.get("cache", "gpu")
    for k in ("train_cams", "test_cams", "eval_cams"):
        out[k] = [cam.to(device) for cam in out[k]]
    for k in ("train_imgs", "test_imgs", "train_masks", "test_masks", "eval_imgs", "eval_masks"):
        if out[k] is not None:
            out[k] = out[k].to(device) if cache == "gpu" or k.startswith("eval") else out[k].pin_memory()
    return out


def target(imgs, masks, i, bg):
    """Float target image: the photo, or the photo composited onto bg with its mask."""
    dev = bg.device
    ref = imgs[i].to(dev, non_blocking=True).float() / 255
    if masks is None:
        return ref, None
    m = masks[i].to(dev, non_blocking=True).float()[..., None] / 255
    return m * ref + (1 - m) * bg, m[..., 0]


@torch.no_grad()
def foreground_points(points, cams, masks, min_inside: float = 0.8, min_views: int = 3):
    """Keep SfM points that project inside the foreground mask in >= ``min_inside`` of the
    training views where they are in frame (and in front of the camera)."""
    from src.render_torch import project_points, world_to_camera
    inside = torch.zeros(len(points), device=points.device)
    seen = torch.zeros(len(points), device=points.device)
    for cam, m in zip(cams, masks):
        pc = world_to_camera(points, cam.viewmat)
        uv = project_points(pc, cam.K)
        u, v = uv[:, 0].long(), uv[:, 1].long()
        ok = (pc[:, 2] > 0.1) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        seen += ok
        m = m.to(points.device, non_blocking=True)        # masks may live in pinned CPU memory
        inside[ok] += (m[v[ok], u[ok]] > 127).float()
    return (seen >= min_views) & (inside >= min_inside * seen.clamp(min=1))


def init_model(cfg, data, device, gen) -> Gaussians:
    m = cfg.model
    sh = m.get("sh_degree", None)
    if m.init == "sfm":
        pts, cols = data["points"].to(device), data["colors"].to(device).float() / 255
        if data["train_masks"] is not None and m.get("filter_points_with_masks", True):
            keep = foreground_points(pts, data["train_cams"], data["train_masks"])
            print(f"SfM init: {int(keep.sum())} of {len(pts)} points are on the foreground")
            pts, cols = pts[keep], cols[keep]
        return Gaussians.from_points(pts, cols, sh_degree=sh, opacity=m.opacity,
                                     scale_mult=m.get("scale_mult", 1.0))
    if m.init == "random_box":
        if m.get("box") is not None:
            lo = torch.full((3,), float(m.box[0]), device=device)
            hi = torch.full((3,), float(m.box[1]), device=device)
        else:       # bounding box of the foreground SfM points, enlarged by box_margin
            pts = data["points"].to(device)
            if data["train_masks"] is not None:
                pts = pts[foreground_points(pts, data["train_cams"], data["train_masks"])]
            lo_q, hi_q = torch.quantile(pts, 0.005, 0), torch.quantile(pts, 0.995, 0)
            pad = (hi_q - lo_q) * m.get("box_margin", 0.1)
            lo, hi = lo_q - pad, hi_q + pad
            print(f"random box init: {lo.tolist()} -> {hi.tolist()}")
        spacing = float(torch.prod(hi - lo)) ** (1 / 3) / m.n ** (1 / 3)
        return Gaussians.random_box(m.n, lo, hi, scale=m.scale_factor * spacing, opacity=m.opacity,
                                    generator=gen, sh_degree=sh)
    raise ValueError(m.init)


# ----------------------------------------------------------------------------- evaluation
class Evaluator:
    """PSNR / SSIM / LPIPS(alex) on the composited test images, plus for masked captures
    PSNR inside the mask only and the IoU of the rendered alpha (> 0.5) with the mask (> 0.5)."""

    def __init__(self, use_lpips: bool, device):
        self.lpips = None
        if use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, render_fn, model, data, bg, idx, sh_degree=None, out_dir=None, save_first=0):
        rows = []
        for j, i in enumerate(idx):
            cam = data["eval_cams"][i]
            img, alpha, _ = render_fn(model, cam, bg, sh_degree)
            img = img.clamp(0, 1)
            ref, m = target(data["eval_imgs"], data["eval_masks"], i, bg)
            r = {"psnr": psnr(img, ref).item(), "ssim": ssim(img, ref).item()}
            if self.lpips is not None:
                to = lambda x: x.permute(2, 0, 1)[None] * 2 - 1
                r["lpips"] = self.lpips(to(img), to(ref)).item()
            if m is not None:
                fg = m > 0.5
                r["psnr_masked"] = (-10 * torch.log10(((img - ref) ** 2)[fg].mean())).item()
                a = alpha > 0.5
                r["alpha_iou"] = ((a & fg).sum() / (a | fg).sum()).item()
            rows.append(r)
            if out_dir is not None and j < save_first:
                pair = torch.cat([ref, img], 1)
                iio.imwrite(out_dir / f"test_{i:03d}.png", (pair * 255 + 0.5).byte().cpu().numpy())
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ----------------------------------------------------------------------------- main
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

    data = load_data(cfg, dev)
    train_cams, test_cams = data["train_cams"], data["test_cams"]
    bg = torch.tensor(list(cfg.data.bg), device=dev)
    render_fn = get_renderer(cfg)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    ckpt_path = run_dir / "ckpt.pt"
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False) \
        if args.resume and ckpt_path.exists() else None
    model = Gaussians.from_state_dict(ck["model"]).to(dev) if ck is not None \
        else init_model(cfg, data, dev, gen)

    # One Adam per parameter tensor (what densification expects: it replaces a tensor when
    # Gaussians are added/removed and rewrites that optimizer's state accordingly).
    if tc.get("optimizer", "adam") == "selective":
        # gsplat's SelectiveAdam: moments/params of Gaussians not visible in this view are not
        # touched (no decay of their moments either) - cheaper for large N.
        from gsplat.optimizers import SelectiveAdam
        make_opt = lambda p, lr, key: SelectiveAdam([{"params": [p], "lr": lr, "name": key}],
                                                    eps=1e-15, betas=(0.9, 0.999))
    else:
        make_opt = lambda p, lr, key: torch.optim.Adam([{"params": [p], "lr": lr, "name": key}], eps=1e-15)
    opts = {key: make_opt(model.params[key], tc.lr[name], key)
            for name, key in LR_KEYS.items() if key in model.params}
    selective = tc.get("optimizer", "adam") == "selective"
    gamma = (tc.means_lr_final / tc.lr.means) ** (1.0 / tc.iterations)
    strategy, strategy_state = make_strategy(cfg, train_cams)
    if strategy is not None:
        strategy.check_sanity(model.params, opts)
    lam_mask = tc.get("lambda_mask", 0.0)
    sh_every = tc.get("sh_every", 1000)
    sh_degree = lambda it: min(it // sh_every, model.sh_degree) if model.uses_sh else None
    evaluator = Evaluator(cfg.get("eval", {}).get("lpips", False), dev)
    eval_idx = np.linspace(0, len(test_cams) - 1, min(cfg.data.eval_subset, len(test_cams))).round().astype(int).tolist()
    print(f"evaluation at {data['eval_cams'][0].width}x{data['eval_cams'][0].height}")
    view_gen = torch.Generator().manual_seed(cfg.seed)
    start, order, elapsed = 0, [], 0.0
    if ck is not None:
        for key, o in opts.items():
            o.load_state_dict(ck["optim"][key])
        if strategy is not None:
            strategy_state = ck["strategy_state"]
        start, order, elapsed = ck["iteration"], ck["order"], ck["elapsed"]
        view_gen.set_state(ck["view_gen"].cpu())       # CPU generator; map_location moved it
        print(f"resumed at iteration {start}")

    def save_ckpt(it):
        tmp = run_dir / "ckpt.pt.tmp"
        torch.save({"model": model.state_dict(), "optim": {k: o.state_dict() for k, o in opts.items()},
                    "strategy_state": strategy_state, "iteration": it, "order": order,
                    "elapsed": elapsed, "view_gen": view_gen.get_state()}, tmp)
        tmp.replace(ckpt_path)          # atomic: an interruption never leaves a torn checkpoint

    print(f"{len(model)} Gaussians ({'SH degree ' + str(model.sh_degree) if model.uses_sh else 'RGB'}), "
          f"{len(train_cams)} train / {len(test_cams)} test views at {train_cams[0].width}x"
          f"{train_cams[0].height}, backend={cfg.render.backend}, masks={data['train_masks'] is not None}, "
          f"lambda_mask={lam_mask}")
    torch.cuda.reset_peak_memory_stats()
    train_peak = 0.0          # training-only peak: the counter is reset after every evaluation
    t_last = time.perf_counter()
    for it in range(start + 1, tc.iterations + 1):
        if not order:
            order = torch.randperm(len(train_cams), generator=view_gen).tolist()
        i = order.pop()
        img, alpha, info = render_fn(model, train_cams[i], bg, sh_degree(it))
        ref, m = target(data["train_imgs"], data["train_masks"], i, bg)
        loss = photometric_loss(img, ref, tc.lambda_dssim)
        if lam_mask > 0 and m is not None:
            loss = loss + lam_mask * (alpha - m).abs().mean()
        if tc.get("opacity_reg", 0) > 0:          # MCMC: L1 on opacity and scale
            loss = loss + tc.opacity_reg * torch.sigmoid(model.params["opacities"]).mean()
        if tc.get("scale_reg", 0) > 0:
            loss = loss + tc.scale_reg * torch.exp(model.params["scales"]).mean()
        if strategy is not None:
            strategy.step_pre_backward(model.params, opts, strategy_state, it, info["meta"])
        loss.backward()
        visible = info["radii"] > 0
        for o in opts.values():
            o.step(visible) if selective else o.step()
            o.zero_grad(set_to_none=True)
        lr_means = tc.lr.means * gamma ** it
        opts["means"].param_groups[0]["lr"] = lr_means
        if strategy is not None:
            # After the optimizer step, as in gsplat's reference trainer: accumulate the
            # screen-space gradient of means2d, then clone/split/prune/reset (ADC) or
            # relocate/add/noise (MCMC, which needs the current position LR).
            kw = {"lr": lr_means} if cfg.get("densify", {}).get("strategy") == "mcmc" else {}
            strategy.step_post_backward(model.params, opts, strategy_state, it, info["meta"],
                                        packed=True, **kw)

        if it % tc.log_every == 0:
            torch.cuda.synchronize()
            train_peak = max(train_peak, torch.cuda.max_memory_allocated() / 2**30)
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
            tb.add_scalar("model/n_gaussians", len(model), it)
            tb.add_scalar("model/opacity_gt_1_255",
                          int((torch.sigmoid(model.params["opacities"]) > 1 / 255).sum()), it)
            print(f"[{it:6d}] loss {loss.item():.4f}  {1e3 * dt:6.0f} ms/it  pairs {info['pairs'] / 1e6:5.1f}M  "
                  f"visible {n_vis:7d}  N {len(model):8d}  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GB",
                  flush=True)
        if it % tc.eval_every == 0 or it == tc.iterations:
            r = evaluator(render_fn, model, data, bg, eval_idx, sh_degree(it))
            for k, v in r.items():
                tb.add_scalar(f"test/{k}", v, it)
            print(f"[{it:6d}] test " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()) +
                  f" (subset of {len(eval_idx)})", flush=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()  # don't bill evaluation memory to training
            t_last = time.perf_counter()        # don't bill evaluation to training time
        if it % tc.ckpt_every == 0 or it == tc.iterations:
            save_ckpt(it)
            t_last = time.perf_counter()

    final = evaluator(render_fn, model, data, bg, range(len(test_cams)), sh_degree(tc.iterations),
                      out_dir=run_dir / "renders", save_first=8)
    o = torch.sigmoid(model.params["opacities"])
    metrics = {**{f"test_{k}": v for k, v in final.items()}, "n_test_views": len(test_cams),
               "resolution": [train_cams[0].width, train_cams[0].height],
               "iterations": tc.iterations, "train_time_s": elapsed,
               "ms_per_iter_mean": 1e3 * elapsed / max(tc.iterations - start, 1) if start == 0 else None,
               "peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30,
               "peak_mem_train_GB": train_peak,
               "n_gaussians": len(model), "n_opacity_gt_1_255": int((o > 1 / 255).sum()),
               "backend": cfg.render.backend, "init": cfg.model.init,
               "colors": f"SH{model.sh_degree}" if model.uses_sh else "RGB",
               "lambda_mask": lam_mask,
               "eval_resolution": [data["eval_cams"][0].width, data["eval_cams"][0].height],
               "antialiased": cfg.render.antialiased,
               "densify": cfg.get("densify", {}).get("strategy", "none")}
    write_metrics(run_dir, metrics)
    tb.close()
    print(json.dumps(metrics, indent=1))
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
