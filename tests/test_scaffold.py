"""Stage 0: config loading and run-directory bookkeeping."""
import json

from omegaconf import OmegaConf

from src.config import load_config
from src.run import make_run_dir, write_metrics


def test_base_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 0\ntrain: {iterations: 30000, lr: 0.01}\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nname: exp\ntrain: {lr: 0.1}\n")
    cfg = load_config(tmp_path / "child.yaml", overrides=["train.iterations=7000"])
    assert cfg.name == "exp"
    assert cfg.seed == 0
    assert cfg.train.lr == 0.1
    assert cfg.train.iterations == 7000
    assert "_base_" not in cfg


def test_run_dir_contents_and_no_overwrite(tmp_path):
    cfg = OmegaConf.create({"name": "smoke", "seed": 1})
    d1, d2 = make_run_dir(cfg, root=tmp_path), make_run_dir(cfg, root=tmp_path)
    assert d1 != d2 and d1.name.endswith("_smoke") and d2.name.endswith("_smoke_1")
    assert (d1 / "tb").is_dir() and (d1 / "renders").is_dir()
    assert OmegaConf.load(d1 / "config.yaml").seed == 1
    assert "commit" in json.loads((d1 / "meta.json").read_text())

    write_metrics(d1, {"psnr": 30.0})
    write_metrics(d1, {"ssim": 0.9})
    assert json.loads((d1 / "metrics.json").read_text()) == {"psnr": 30.0, "ssim": 0.9}
