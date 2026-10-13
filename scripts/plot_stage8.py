"""Stage 8 analysis: the resolution x cap x densification table and full-res crops.

    python -m scripts.plot_stage8
All runs are scored on the same 27 held-out photos at full resolution (3024x4032); the best
run is the one with the lowest LPIPS. Stage 7 D is included as the 1600 px baseline
(re-scored at full resolution with src/evaluate.py -> metrics_full.json).
"""
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
INK, MUTED = "#0b0b0b", "#52514e"


def describe(run: Path) -> dict:
    cfg = OmegaConf.load(run / "config.yaml")
    d = cfg.get("densify", {})
    method = {"adc": "ADC (3DGS)", "mcmc": "MCMC", "none": "none"}[d.get("strategy", "none")]
    if d.get("strategy") == "adc" and d.get("absgrad", False):
        method = f"AbsGS (τ={d.get('grow_grad2d')})"
    cap = f"{d.get('cap_max', 0) / 1e6:.0f}M" if d.get("strategy") == "mcmc" else "—"
    return {"method": method, "res": str(cfg.data.res), "cap": cap,
            "aa": bool(cfg.render.antialiased), "iters": int(cfg.train.iterations)}


def collect():
    rows = []
    d7 = sorted((ROOT / "outputs").glob("*stage7_D_sh"))[-1]
    for run in [d7] + sorted((ROOT / "outputs").glob("*stage8_*")):
        mfile = run / ("metrics_full.json" if run == d7 else "metrics.json")
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text())
        tm = json.loads((run / "metrics.json").read_text())          # training stats
        get = lambda k: m.get(f"test_{k}", m.get(k))
        if (m.get("eval_resolution") or m.get("resolution")) != [3024, 4032]:
            continue
        rows.append({"run": run, **describe(run), "psnr": get("psnr"), "psnr_masked": get("psnr_masked"),
                     "ssim": get("ssim"), "lpips": get("lpips"), "n": tm["n_gaussians"],
                     "vram": tm.get("peak_mem_train_GB") or tm["peak_mem_GB"],
                     "vram_incl_eval": "peak_mem_train_GB" not in tm,
                     "minutes": tm["train_time_s"] / 60,
                     "it_s": 1e3 / tm["ms_per_iter_mean"] if tm.get("ms_per_iter_mean") else None})
    return rows


def main():
    rows = collect()
    best = min(rows, key=lambda r: r["lpips"])
    rows.sort(key=lambda r: ({"1600": 0, "2000": 1, "full": 2}[r["res"]], r["lpips"]))
    print("| train res | densification | cap | AA | iters | PSNR | PSNR (plant) | SSIM | LPIPS ↓ | Gaussians | "
          "peak VRAM | it/s | time |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        star = " **(best)**" if r is best else ""
        res = {"1600": "1200×1600", "2000": "1500×2000", "full": "3024×4032"}[r["res"]]
        its = f"{r['it_s']:.1f}" if r["it_s"] else "—"
        print(f"| {res} | {r['method']} | {r['cap']} | {'✓' if r['aa'] else ''} | {r['iters'] // 1000}k | "
              f"{r['psnr']:.2f} | {r['psnr_masked']:.2f} | {r['ssim']:.4f} | {r['lpips']:.4f}{star} | "
              f"{r['n'] / 1e3:.0f}k | {r['vram']:.1f} GB{'*' if r['vram_incl_eval'] else ''} | {its} | {r['minutes']:.0f} min |")
    print(f"\nbest (LPIPS): {best['run'].name}")

    # ---- full-res crops: GT + selected runs, one test view, two detail windows
    pick = []
    for want in (("1600", "ADC (3DGS)", False), ("1600", "ADC (3DGS)", True), ("2000", "ADC (3DGS)", False),
                 ("2000", "MCMC", False), ("full", "ADC (3DGS)", False)):
        c = [r for r in rows if (r["res"], r["method"], r["aa"]) == want and r["iters"] == 30000]
        if c:
            pick.append(c[0])
    if best not in pick:
        pick.append(best)
    view = "test_002.png"
    def load(r):
        d = r["run"] / ("renders_full" if (r["run"] / "renders_full" / view).exists() else "renders")
        im = iio.imread(d / view); w = im.shape[1] // 2
        return im[:, :w], im[:, w:]
    gt, _ = load(pick[0])
    H, W = gt.shape[:2]
    g = np.abs(np.diff(gt.mean(-1), axis=0))[:, :-1] + np.abs(np.diff(gt.mean(-1), axis=1))[:-1]
    s = 360
    cand = sorted(((g[y:y + s, x:x + s].sum(), y, x) for y in range(0, H - s, 120) for x in range(0, W - s, 120)),
                  reverse=True)
    wins = [cand[0][1:]]
    for _, y, x in cand[1:]:
        if all(abs(y - yy) > s or abs(x - xx) > s for yy, xx in wins):
            wins.append((y, x)); break
    cols = [("ground truth (full res)", gt)] + [
        (f"{r['res']} · {r['method']}{' · AA' if r['aa'] else ''} · {r['iters'] // 1000}k\nLPIPS {r['lpips']:.4f}",
         load(r)[1]) for r in pick]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.9 * len(cols), 6.4), dpi=160)
    for c, (t, img) in enumerate(cols):
        for r_, (y, x) in enumerate(wins):
            axes[r_, c].imshow(img[y:y + s, x:x + s], interpolation="nearest")
            axes[r_, c].set_xticks([]); axes[r_, c].set_yticks([])
        axes[0, c].set_title(t, fontsize=7.5, color=INK)
    fig.suptitle("Full-resolution crops (360×360 px of the 3024×4032 test photo)", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "stage8_crops.png"); plt.close(fig)

    # ---- LPIPS vs training time
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    colors = {"1600": "#9a9994", "2000": "#2a78d6", "full": "#eb6834"}
    for r in rows:
        ax.scatter(r["minutes"], r["lpips"], s=40, color=colors[r["res"]], zorder=3,
                   marker="o" if "MCMC" not in r["method"] else "s", edgecolor="white", lw=0.8)
        ax.annotate(f"{r['method'].split(' ')[0]}{' AA' if r['aa'] else ''}{' 60k' if r['iters'] > 30000 else ''}",
                    (r["minutes"], r["lpips"]), xytext=(5, 3), textcoords="offset points", fontsize=7, color=MUTED)
    for res, col in colors.items():
        ax.scatter([], [], color=col, label=f"trained at {res}")
    ax.set_xlabel("training time (min)", color=MUTED); ax.set_ylabel("LPIPS at full res (lower is better)", color=MUTED)
    ax.legend(frameon=False, fontsize=8); ax.grid(color="#e4e3df")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "stage8_lpips_vs_time.png"); plt.close(fig)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
