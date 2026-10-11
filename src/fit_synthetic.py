"""Stage 3: recover known 3D Gaussians from rendered views.

    python -m src.fit_synthetic init.mode=random seed=1
    python -m src.fit_synthetic init.mode=gt_perturbed

Ground truth: ``make_gt_scene`` rendered with ``render_torch`` from cameras on a sphere.
The optimized Gaussians are rendered with the *same* renderer, so with a perfect
optimizer the problem is exactly solvable; any residual error is optimization
(local minima) or identifiability, not model mismatch.
Writes gt.pt / final.pt (activated parameters), metrics.json, TensorBoard logs,
renders/ (test views, GT vs recovered) and progress.gif (one test view over time).
"""
from __future__ import annotations

import argparse
import json
import time

import imageio.v3 as iio
import numpy as np
import torch

from src.camera import sphere_cameras
from src.config import load_config
from src.gaussians import Gaussians
from src.render_torch import render
from src.run import REPO_ROOT, make_run_dir, seed_everything, write_metrics
from src.synthetic import compare_parameters, make_gt_scene, summarize


def psnr(a, b):
    return float(-10 * torch.log10(torch.mean((a.clamp(0, 1) - b) ** 2)))


def to_u8(img):
    return (img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def init_model(cfg, gt, dev, gen) -> Gaussians:
    ic = cfg.init
    if ic.mode == "gt_perturbed":
        model = Gaussians.from_activated(gt["means"], gt["quats"], gt["scales"],
                                         gt["opacities"], gt["colors"])
        with torch.no_grad():
            for name, std in ic.perturb.items():
                p = getattr(model, name)
                p.add_(std * torch.randn(p.shape, generator=gen, device=dev))
        return model
    if ic.mode == "random":
        e = cfg.scene.extent
        lo, hi = torch.full((3,), -e, device=dev), torch.full((3,), e, device=dev)
        return Gaussians.random_box(ic.n, lo, hi, opacity=ic.opacity, generator=gen)
    raise ValueError(ic.mode)


@torch.no_grad()
def relocate_dead(model: Gaussians, opt: torch.optim.Optimizer, min_opacity: float,
                  gen: torch.Generator) -> int:
    """Move dead Gaussians (opacity < min_opacity) onto live ones (3DGS-MCMC style).

    Targets are sampled with probability proportional to opacity. A target receiving k
    copies keeps its geometry and color, and all k+1 copies get opacity
    o' = 1 - (1 - o)^(1/(k+1)), so that stacking them reproduces the original alpha at the
    center: 1 - (1 - o')^(k+1) = o. (MCMC-3DGS also shrinks the scales; omitted here.)
    Adam moments of the moved Gaussians are reset. Returns the number relocated.
    """
    o = torch.sigmoid(model.opacity_logits)
    dead = torch.nonzero(o < min_opacity).squeeze(1)
    alive = torch.nonzero(o >= min_opacity).squeeze(1)
    if len(dead) == 0 or len(alive) == 0:
        return 0
    pick = alive[torch.multinomial(o[alive], len(dead), replacement=True, generator=gen)]
    copies = torch.bincount(pick, minlength=len(o)).float()
    new_o = 1 - (1 - o) ** (1.0 / (copies + 1))
    for name in ("means", "quats", "log_scales", "color_logits"):
        getattr(model, name)[dead] = getattr(model, name)[pick]
    logits = torch.log(new_o / (1 - new_o)).clamp(-20, 20)
    model.opacity_logits[pick] = logits[pick]
    model.opacity_logits[dead] = logits[pick]
    for group in opt.param_groups:
        st = opt.state.get(group["params"][0], {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key][dead] = 0
    return len(dead)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage3_synthetic.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / args.config, args.overrides)
    if not any(o.startswith("name=") for o in args.overrides):
        tag = "gtpert" if cfg.init.mode == "gt_perturbed" else f"random_n{cfg.init.n}"
        if cfg.train.get("relocate_every", 0):
            tag += "_reloc"
        cfg.name = f"{cfg.name}_{tag}_s{cfg.seed}"
    seed_everything(cfg.seed)
    dev = torch.device(cfg.device)
    run_dir = make_run_dir(cfg)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(run_dir / "tb")

    # ---- ground truth scene and images
    sc, cc, tc = cfg.scene, cfg.cameras, cfg.train
    gt = make_gt_scene(sc.n_gt, sc.extent, tuple(sc.scale_range), tuple(sc.opacity_range),
                       seed=sc.seed, device=dev)
    cams = sphere_cameras(cc.n, cc.radius, cc.fov_deg, cc.width, cc.height, seed=sc.seed, device=dev)
    test_idx = list(range(0, cc.n, cc.test_every))
    train_idx = [i for i in range(cc.n) if i not in test_idx]
    bg = torch.tensor(list(cc.bg), device=dev)
    W, H = cc.width, cc.height
    gt_args = (gt["means"], gt["quats"], gt["scales"], gt["opacities"], gt["colors"])
    with torch.no_grad():
        gt_imgs = torch.stack([render(*gt_args, c.viewmat, c.K, W, H, bg=bg)[0] for c in cams])
    torch.save({k: v.cpu() for k, v in gt.items()}, run_dir / "gt.pt")

    # ---- model and optimizer
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    model = init_model(cfg, gt, dev, gen)
    opt = torch.optim.Adam([{"params": [getattr(model, k)], "lr": v, "name": k}
                            for k, v in tc.lr.items()], eps=1e-15)
    gamma = tc.means_lr_final_factor ** (1.0 / tc.iterations)
    means_group = next(g for g in opt.param_groups if g["name"] == "means")

    def render_view(i):
        c = cams[i]
        return render(*model.activated(), c.viewmat, c.K, W, H, bg=bg)

    @torch.no_grad()
    def evaluate():
        return float(np.mean([psnr(render_view(i)[0], gt_imgs[i]) for i in test_idx]))

    def param_metrics():
        m, q, s, o, c = model.activated()
        rec = {"means": m, "quats": q, "scales": s, "opacities": o, "colors": c}
        return compare_parameters(gt, rec)

    gif_view = test_idx[1]
    frames = []
    view_gen = torch.Generator().manual_seed(cfg.seed)
    history = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for it in range(0, tc.iterations + 1):
        if it % tc.eval_every == 0:
            p = evaluate()
            tb.add_scalar("test/psnr", p, it)
            row = {"it": it, "test_psnr": p, "time_s": time.perf_counter() - t0}
            if it % tc.params_every == 0:
                s = summarize(param_metrics())
                for k, v in s.items():
                    if v is not None:
                        tb.add_scalar(f"params/{k}", v, it)
                row.update(s)
                print(f"[{it:5d}] test PSNR {p:6.2f}  recovered {s['recovered_frac']:.2f}  "
                      f"pos {s['pos_err_median']:.4f}  covrel {s['cov_rel_err_median']:.3f}  "
                      f"{time.perf_counter() - t0:6.1f}s", flush=True)
            history.append(row)
        if it % tc.gif_every == 0:
            with torch.no_grad():
                frames.append(to_u8(torch.cat([gt_imgs[gif_view], render_view(gif_view)[0]], 1)))
        if it == tc.iterations:
            break

        views = torch.randperm(len(train_idx), generator=view_gen)[:tc.views_per_iter]
        loss = 0.0
        for v in views.tolist():
            i = train_idx[v]
            img = render_view(i)[0]
            err = img - gt_imgs[i]
            loss = loss + (err.abs().mean() if tc.loss == "l1" else (err ** 2).mean())
        loss = loss / len(views)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        means_group["lr"] *= gamma
        re = tc.get("relocate_every", 0)
        if re and it > 0 and it % re == 0 and it <= tc.get("relocate_until", tc.iterations // 2):
            n_moved = relocate_dead(model, opt, tc.get("relocate_min_opacity", 0.005), gen)
            tb.add_scalar("train/relocated", n_moved, it)
        if it % tc.eval_every == 0:
            tb.add_scalar("train/loss", loss.item(), it)
    train_time = time.perf_counter() - t0

    # ---- final evaluation
    m = param_metrics()
    s = summarize(m)
    with torch.no_grad():
        for j, i in enumerate(test_idx[:4]):
            iio.imwrite(run_dir / "renders" / f"test{j}.png",
                        to_u8(torch.cat([gt_imgs[i], render_view(i)[0]], 1)))
    iio.imwrite(run_dir / "progress.gif", np.stack(frames), duration=80, loop=0)
    act = model.activated()
    torch.save({k: v.detach().cpu() for k, v in
                zip(("means", "quats", "scales", "opacities", "colors"), act)}, run_dir / "final.pt")
    np.savez(run_dir / "param_errors.npz", **{k: v for k, v in m.items() if isinstance(v, np.ndarray)})
    (run_dir / "history.json").write_text(json.dumps(history))
    metrics = {"test_psnr": evaluate(), "train_time_s": train_time,
               "ms_per_iter": 1e3 * train_time / tc.iterations,
               "peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30,
               "init": cfg.init.mode, "n_init": len(model), "n_gt": sc.n_gt, **s}
    write_metrics(run_dir, metrics)
    tb.close()
    print(json.dumps(metrics, indent=1))
    print(f"-> {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
