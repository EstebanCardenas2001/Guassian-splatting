"""Experiment configuration: one YAML per experiment, optional CLI overrides.

Configs may inherit from another config with a top-level ``_base_`` key (path
relative to the child file). Merging is recursive; the child wins.

Example::

    cfg = load_config("configs/lego.yaml", overrides=["train.iterations=7000"])
    cfg.train.iterations  # -> 7000
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Load ``path`` (resolving ``_base_`` chains) and apply ``key=value`` dotlist overrides."""
    path = Path(path).resolve()
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is not None:
        cfg = OmegaConf.merge(load_config(path.parent / base), cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
