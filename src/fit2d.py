"""Stage 1: fit an image with N 2D Gaussians.

    python -m src.fit2d --config configs/stage1_fit2d.yaml model.n=5000 model.mode=additive

Raw (unconstrained) parameters and their activations:
    means           (N, 2)  identity, pixels
    theta           (N,)    identity, radians
    log_scales      (N, 2)  sigma = exp(.)
    color_logits    (N, 3)  c = sigmoid(.)
    opacity_logits  (N,)    o = sigmoid(.)
"""
from __future__ import annotations

import argparse
import math
import time

import imageio.v3 as iio
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.run import REPO_ROOT, make_run_dir, seed_everything, write_metrics
from src.splat2d import render


def load_image(path: str, crop: int, device) -> torch.Tensor:
    """Read an image, center-crop to ``crop`` x ``crop``, return float (H, W, 3) in [0, 1]."""
    img = iio.imread(REPO_ROOT / path)[..., :3]
    H, W = img.shape[:2]
    if crop and crop < min(H, W):
        y0, x0 = (H - crop) // 2, (W - crop) // 2
        img = img[y0:y0 + crop, x0:x0 + crop]
    return torch.from_numpy(np.ascontiguousarray(img)).to(device).float() / 255.0


def logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


class Gaussians2D(nn.Module):
    def __init__(self, target: torch.Tensor, n: int, init: str, scale_init: float,
                 opacity_init: float, generator: torch.Generator):
        super().__init__()
        H, W, _ = target.shape
        dev = target.device
        if init == "uniform":
            means = torch.rand(n, 2, generator=generator, device=dev) * torch.tensor([W, H], device=dev)
        elif init == "gradient":
            # Sample pixel locations with probability ∝ |∇I| + mean|∇I| (half uniform, half edges).
            gray = target.mean(-1)
            gx = torch.zeros_like(gray); gy = torch.zeros_like(gray)
            gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
            mag = torch.sqrt(gx**2 + gy**2).flatten()
            idx = torch.multinomial(mag + mag.mean(), n, replacement=True, generator=generator)
            jitter = torch.rand(n, 2, generator=generator, device=dev)
            means = torch.stack([idx % W, idx // W], -1).float() + jitter
        else:
            raise ValueError(init)
        sigma0 = scale_init * math.sqrt(H * W / n)
        pix = means.long()
        self.means = nn.Parameter(means)
        self.theta = nn.Parameter(torch.rand(n, generator=generator, device=dev) * math.pi)
        self.log_scales = nn.Parameter(torch.full((n, 2), math.log(sigma0), device=dev))
        self.color_logits = nn.Parameter(logit(target[pix[:, 1], pix[:, 0]]))
        self.opacity_logits = nn.Parameter(logit(torch.full((n,), opacity_init, device=dev)))

    def activated(self):
        return (self.means, self.theta, torch.exp(self.log_scales),
                torch.sigmoid(self.color_logits), torch.sigmoid(self.opacity_logits))


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.clamp(0, 1) - b) ** 2)
    return float(-10 * torch.log10(mse))


def to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_fit2d.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args(argv)
    cfg = load_config(REPO_ROOT / args.config, args.overrides)
    if not any(o.startswith("name=") for o in args.overrides):
        cfg.name = f"{cfg.name}_{cfg.model.mode}_n{cfg.model.n}"
    seed_everything(cfg.seed)
    dev = torch.device(cfg.device)
    run_dir = make_run_dir(cfg)
    tb = SummaryWriter(run_dir / "tb")

    target = load_image(cfg.image.path, cfg.image.crop, dev)
    H, W, _ = target.shape
    iio.imwrite(run_dir / "renders" / "target.png", to_uint8(target))
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    m, tcfg = cfg.model, cfg.train
    model = Gaussians2D(target, m.n, m.init, m.scale_init, m.opacity_init, gen)
    bg = torch.tensor(list(m.bg), device=dev)

    opt = torch.optim.Adam([{"params": [getattr(model, k)], "lr": v, "name": k}
                            for k, v in tcfg.lr.items()], eps=1e-15)
    gamma = tcfg.lr_final_factor ** (1.0 / tcfg.iterations)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)

    def forward():
        return render(*model.activated(), H, W, mode=m.mode, bg=bg,
                      t_min=m.t_min, max_radius=m.max_radius)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(1, tcfg.iterations + 1):
        img, stats = forward()
        err = img - target
        loss = (err ** 2).mean() if tcfg.loss == "l2" else err.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if it % tcfg.log_every == 0 or it == 1:   # the only GPU->CPU syncs in the loop
            p = psnr(img.detach(), target)
            tb.add_scalar("train/loss", loss.item(), it)
            tb.add_scalar("train/psnr", p, it)
            tb.add_scalar("stats/pairs_M", stats["pairs"] / 1e6, it)
            tb.add_scalar("stats/sec", time.perf_counter() - t0, it)
            if it % (tcfg.log_every * 10) == 0:
                print(f"[{it:5d}] loss {loss.item():.5f}  psnr {p:6.2f}  "
                      f"pairs {stats['pairs'] / 1e6:5.2f}M  {time.perf_counter() - t0:6.1f}s",
                      flush=True)
        if it % tcfg.render_every == 0:
            iio.imwrite(run_dir / "renders" / f"it{it:05d}.png", to_uint8(img.detach()))
    torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    with torch.no_grad():
        img, stats = forward()
    final = psnr(img, target)
    iio.imwrite(run_dir / "renders" / "final.png", to_uint8(img))
    metrics = {"psnr": final, "n": m.n, "mode": m.mode, "init": m.init,
               "iterations": tcfg.iterations, "train_time_s": train_time,
               "ms_per_iter": 1e3 * train_time / tcfg.iterations,
               "pairs_final": stats["pairs"], "H": H, "W": W,
               "peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30}
    write_metrics(run_dir, metrics)
    tb.add_hparams({"n": m.n, "mode": m.mode, "init": m.init}, {"hparam/psnr": final})
    tb.close()
    print(f"done: PSNR {final:.2f} dB  ({metrics['ms_per_iter']:.1f} ms/it, "
          f"peak {metrics['peak_mem_GB']:.2f} GB)  -> {run_dir}")
    return metrics


if __name__ == "__main__":
    main()
