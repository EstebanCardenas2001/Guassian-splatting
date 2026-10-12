"""Evaluate a trained run on its held-out views, at any resolution.

    python -m src.evaluate --run outputs/<run> [--res full] [--save 8]

Loads the run's config and checkpoint, renders the test split at ``--res`` (default: the
run's eval resolution) and writes ``metrics_<res>.json`` (PSNR, SSIM, LPIPS-alex, PSNR
inside the mask, alpha IoU, render time) plus side-by-side renders in ``renders_<res>/``.
Rendering a model at a higher resolution than it was trained at is legitimate: the
Gaussians are continuous, and the full-resolution photos are the common ground truth on
which runs trained at 1600 px, 2000 px and full resolution are compared.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.camera import load_colmap
from src.gaussians import Gaussians
from src.run import REPO_ROOT
from src.train import Evaluator, get_renderer


def evaluate_run(run_dir: Path, res: str | None = None, save: int = 8) -> dict:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    d = cfg.data
    res = str(res or cfg.get("eval", {}).get("res", None) or d.res)
    dev = torch.device("cuda")
    ck = torch.load(run_dir / "ckpt.pt", map_location=dev, weights_only=False)
    model = Gaussians.from_state_dict(ck["model"]).to(dev)
    e = load_colmap(REPO_ROOT / d.path, res=res, model=d.model, test_every=d.test_every, load_split="test")
    data = {"eval_cams": [c.to(dev) for c in e["test_cams"]], "eval_imgs": e["test_images"].to(dev),
            "eval_masks": e["test_masks"].to(dev) if d.use_masks else None}
    bg = torch.tensor(list(d.bg), device=dev)
    render_fn = get_renderer(cfg)
    out_dir = run_dir / f"renders_{res}"
    out_dir.mkdir(exist_ok=True)
    ev = Evaluator(True, dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = ev(render_fn, model, data, bg, range(len(data["eval_cams"])),
           sh_degree=model.sh_degree if model.uses_sh else None, out_dir=out_dir, save_first=save)
    torch.cuda.synchronize()
    cam = data["eval_cams"][0]
    r.update(resolution=[cam.width, cam.height], n_views=len(data["eval_cams"]), n_gaussians=len(model),
             eval_seconds=time.perf_counter() - t0)
    (run_dir / f"metrics_{res}.json").write_text(json.dumps(r, indent=1))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--res")
    ap.add_argument("--save", type=int, default=8)
    args = ap.parse_args()
    for run in args.run:
        r = evaluate_run(Path(run), args.res, args.save)
        print(run, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))


if __name__ == "__main__":
    main()
