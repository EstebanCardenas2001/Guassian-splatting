"""Training-only peak VRAM of finished runs, measured after the fact.

    python -m scripts.measure_train_peak outputs/<run> [...]
Runs whose metrics.json predates ``peak_mem_train_GB`` report a peak that includes the
periodic full-resolution evaluation. This replays 10 full training steps (render at the
run's training resolution with its settings, L1+SSIM+mask loss, backward, Adam on every
parameter) from the run's *final* checkpoint - the largest the model ever was, since all
densification here grows N until 15-25k iterations - and writes the peak into metrics.json
as ``peak_mem_train_GB`` (with ``peak_mem_train_source: "replay"``).
"""
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.camera import load_colmap
from src.gaussians import Gaussians
from src.losses import photometric_loss
from src.run import REPO_ROOT
from src.train import get_renderer


def replay(run: Path) -> float:
    cfg = OmegaConf.load(run / "config.yaml")
    d = cfg.data
    model = Gaussians.from_state_dict(torch.load(run / "ckpt.pt", map_location="cuda", weights_only=False)["model"]).cuda()
    opts = [torch.optim.Adam([p], lr=1e-5, eps=1e-15) for p in model.params.values()]
    e = load_colmap(REPO_ROOT / d.path, res=str(d.res), model=d.model, test_every=d.test_every, load_split="test")
    render_fn = get_renderer(cfg)
    bg = torch.tensor(list(d.bg), device="cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for i in range(10):
        cam = e["test_cams"][i % len(e["test_cams"])].to("cuda")
        img, alpha, _ = render_fn(model, cam, bg, model.sh_degree)
        ref = e["test_images"][i % len(e["test_cams"])].cuda().float() / 255
        m = e["test_masks"][i % len(e["test_cams"])].cuda().float() / 255
        ref = m[..., None] * ref + (1 - m[..., None]) * bg
        loss = photometric_loss(img, ref, cfg.train.lambda_dssim) + cfg.train.get("lambda_mask", 0) * (alpha - m).abs().mean()
        loss.backward()
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    for run in map(Path, sys.argv[1:]):
        mf = run / "metrics.json"
        m = json.loads(mf.read_text())
        if "peak_mem_train_GB" in m:
            print(f"{run.name}: already has peak_mem_train_GB = {m['peak_mem_train_GB']:.2f}")
            continue
        peak = replay(run)
        m.update(peak_mem_train_GB=peak, peak_mem_train_source="replay of 10 steps from the final checkpoint")
        mf.write_text(json.dumps(m, indent=1))
        print(f"{run.name}: training peak {peak:.2f} GB (was {m['peak_mem_GB']:.2f} incl. evaluation)", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
