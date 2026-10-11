"""Run bookkeeping: seeding, git provenance and the ``outputs/<date>_<name>/`` directory.

Every run directory contains:
    config.yaml   the fully merged config that was actually used
    meta.json     git commit, dirty flag, start time, host, torch/CUDA versions
    metrics.json  written by the training/eval code (see ``write_metrics``)
    tb/           TensorBoard event files
    renders/      validation images
"""
from __future__ import annotations

import datetime as dt
import json
import random
import socket
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch on CPU and all GPUs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_info(repo: Path = REPO_ROOT) -> dict:
    """Return ``{"commit": sha | None, "dirty": bool | None}`` for ``repo``."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def make_run_dir(cfg: DictConfig, root: Path | None = None) -> Path:
    """Create ``<root>/<YYYY-MM-DD>_<cfg.name>[_k]/`` and write config.yaml + meta.json.

    A numeric suffix is appended if the directory already exists, so runs are
    never overwritten.
    """
    root = Path(root) if root is not None else REPO_ROOT / cfg.get("output_root", "outputs")
    stem = f"{dt.date.today().isoformat()}_{cfg.name}"
    run_dir, k = root / stem, 1
    while run_dir.exists():
        run_dir, k = root / f"{stem}_{k}", k + 1
    (run_dir / "tb").mkdir(parents=True)
    (run_dir / "renders").mkdir()

    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    meta = {"start_time": dt.datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(), **git_info()}
    try:
        import torch
        meta |= {"torch": torch.__version__, "cuda": torch.version.cuda,
                 "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    except ImportError:
        pass
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def write_metrics(run_dir: Path, metrics: dict) -> None:
    """Merge ``metrics`` into ``run_dir/metrics.json``."""
    path = Path(run_dir) / "metrics.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(current | metrics, indent=2))
