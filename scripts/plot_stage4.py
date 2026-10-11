"""Stage 4 figures: lego training curves, test renders, and the pure-PyTorch cost model.

    python -m scripts.plot_stage4 --run outputs/<lego run> --profiles /mnt/outputs/profiles/*.json
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VRAM_GB = 15.6 * 0.93    # usable on the T4 after the CUDA context


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def scalars(run, tag):
    ea = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0}); ea.Reload()
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--profiles", nargs="+", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- training curves
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), dpi=150)
    it, p = scalars(run, "test/psnr")
    axes[0].plot(it, p, color=BLUE, lw=2, marker="o", ms=4)
    axes[0].set_ylabel("test PSNR (dB, 20-view subset)", color=MUTED)
    it, ms = scalars(run, "perf/ms_per_iter")
    axes[1].plot(it, ms, color=ORANGE, lw=1.5)
    axes[1].set_ylabel("ms / iteration", color=MUTED)
    it, pairs = scalars(run, "perf/pairs_M")
    it2, alive = scalars(run, "model/opacity_gt_1_255")
    axes[2].plot(it, pairs, color=AQUA, lw=1.5, label="pairs per view (M)")
    axes[2].plot(it2, alive / 1e5, color=MUTED, lw=1.5, ls="--", label="Gaussians with o > 1/255 (×100k)")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("iteration", color=MUTED); style(ax)
    fig.tight_layout(); fig.savefig(OUT / "stage4_training.png"); plt.close(fig)

    # ---- test renders: 4 views, GT | render (files hold the pair side by side)
    files = sorted((run / "renders").glob("test_*.png"))[:4]
    fig, axes = plt.subplots(2, len(files), figsize=(3.2 * len(files), 6.6), dpi=150)
    for j, f in enumerate(files):
        im = iio.imread(f); w = im.shape[1] // 2
        for r, (part, lab) in enumerate(((im[:, :w], "ground truth"), (im[:, w:], "render (pure PyTorch)"))):
            axes[r, j].imshow(part); axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if j == 0:
                axes[r, j].set_ylabel(lab, fontsize=9, color=INK)
        axes[0, j].set_title(f.stem.replace("_", " "), fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "stage4_renders.png"); plt.close(fig)

    # ---- cost model: time and memory vs pairs, across all profiled configurations
    rows = []
    for f in args.profiles:
        src = "trained" if "trained" in Path(f).name else "random init"
        for r in json.load(open(f)):
            r["src"] = src; rows.append(r)
    ok = [r for r in rows if r["status"] == "ok"]
    x = np.array([r["pairs_M"] for r in ok]); t = np.array([r["ms"] for r in ok]); m = np.array([r["peak_GB"] for r in ok])
    kt = np.polyfit(x, t, 1); km = np.polyfit(x, m, 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    for ax, y, k, lab in ((axes[0], t, kt, "ms per forward+backward"), (axes[1], m, km, "peak VRAM (GB)")):
        for src, col in (("random init", ORANGE), ("trained", BLUE)):
            sel = np.array([r["src"] == src for r in ok])
            if sel.any():
                ax.scatter(x[sel], y[sel], s=18, color=col, label=src, zorder=3)
        xx = np.linspace(0, max(x.max(), (VRAM_GB - km[1]) / km[0]) * 1.05, 50)
        ax.plot(xx, np.polyval(k, xx), color=MUTED, lw=1, ls="--", zorder=2,
                label=f"fit: {k[0]:.3g}·P + {k[1]:.2g}")
        ax.set_xlabel("(Gaussian, pixel) pairs P (millions)", color=MUTED); ax.set_ylabel(lab, color=MUTED); style(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[1].axhline(VRAM_GB, color=ORANGE, lw=1)
    axes[1].annotate("usable T4 memory", (0, VRAM_GB), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "stage4_cost_model.png"); plt.close(fig)
    p_max = (VRAM_GB - km[1]) / km[0]
    print(f"time ≈ {kt[0]:.1f} ms/Mpair + {kt[1]:.0f} ms; memory ≈ {1e3 * km[0]:.0f} B/pair "
          f"+ {km[1]:.2f} GB -> memory limit at ≈ {p_max:.0f} M pairs")

    # ---- table of profiled configurations
    print("\n| state | resolution | N | init scale | pairs (M) | ms / iter | peak VRAM (GB) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        sf = r.get("scale_factor", "")
        if r["status"] == "ok":
            print(f"| {r['src']} | {r['res']}² | {r['n']:,} | {sf} | {r['pairs_M']:.1f} | {r['ms']:.0f} | {r['peak_GB']:.2f} |")
        else:
            reason = "OOM" if "OOM" in r["status"] else "> 80 M pairs (cap)"
            print(f"| {r['src']} | {r['res']}² | {r['n']:,} | {sf} | — | {reason} | — |")
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
