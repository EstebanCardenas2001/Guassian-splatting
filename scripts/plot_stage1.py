"""Collect Stage 1 runs (outputs/*stage1_fit2d_*/metrics.json) into a PSNR-vs-N plot,
a markdown table and a render comparison panel. Writes to docs/figures/ and prints the table.

    python scripts/plot_stage1.py [--date 2026-10-11]
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"additive": "#2a78d6", "alpha": "#eb6834"}   # categorical slots 1, 2
LABELS = {"additive": "additive", "alpha": "alpha compositing"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def collect(date: str | None):
    runs = {}
    for f in sorted((ROOT / "outputs").glob("*stage1_fit2d_*/metrics.json")):
        if date and not f.parent.name.startswith(date):
            continue
        m = json.loads(f.read_text())
        runs[(m["mode"], m["n"])] = (m, f.parent)   # later runs of the same config win
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    args = ap.parse_args()
    runs = collect(args.date)
    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for mode in ("additive", "alpha"):
        pts = sorted((n, m["psnr"]) for (md, n), (m, _) in runs.items() if md == mode)
        if not pts:
            continue
        ns, ps = zip(*pts)
        ax.plot(ns, ps, color=COLORS[mode], lw=2, marker="o", ms=7,
                markeredgecolor="white", markeredgewidth=2, label=LABELS[mode], zorder=3)
        ax.annotate(LABELS[mode], (ns[-1], ps[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.axhline(30, color=MUTED, lw=1, ls="--", zorder=1)
    ax.annotate("30 dB target", (ax.get_xlim()[0], 30), xytext=(4, 4), textcoords="offset points",
                color=MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([1000, 5000, 20000], ["1k", "5k", "20k"])
    ax.minorticks_off()
    ax.set_xlabel("number of Gaussians N", color=MUTED)
    ax.set_ylabel("PSNR (dB)", color=MUTED)
    ax.set_title("Stage 1: 2D Gaussian fit of a 512² crop", color=INK, loc="left", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim(700, 40000)
    fig.tight_layout()
    fig.savefig(out / "stage1_psnr_vs_n.png")

    # Table
    print("| blending | N | PSNR (dB) | ms/iter | pairs (M) | peak VRAM (GB) | run |")
    print("|---|---|---|---|---|---|---|")
    for (mode, n), (m, d) in sorted(runs.items()):
        print(f"| {mode} | {n} | {m['psnr']:.2f} | {m['ms_per_iter']:.1f} | "
              f"{m['pairs_final'] / 1e6:.2f} | {m['peak_mem_GB']:.2f} | {d.name} |")

    # Render panel: target + each N for alpha compositing, plus a zoomed crop row.
    alpha_runs = sorted((n, d) for (md, n), (_, d) in runs.items() if md == "alpha")
    if alpha_runs:
        tgt = iio.imread(alpha_runs[0][1] / "renders" / "target.png")
        imgs = [("target", tgt)] + [(f"N = {n // 1000}k", iio.imread(d / "renders" / "final.png"))
                                     for n, d in alpha_runs]
        H = tgt.shape[0]
        y0, x0, s = H // 8, H // 3, H // 4            # zoom window
        fig, axes = plt.subplots(2, len(imgs), figsize=(3 * len(imgs), 6), dpi=150)
        for j, (title, im) in enumerate(imgs):
            axes[0, j].imshow(im)
            axes[0, j].set_title(title, fontsize=10, color=INK)
            axes[1, j].imshow(im[y0:y0 + s, x0:x0 + s], interpolation="nearest")
            for a in axes[:, j]:
                a.set_xticks([]); a.set_yticks([])
        axes[1, 0].set_ylabel(f"zoom {s}×{s} px", fontsize=9, color=MUTED)
        fig.tight_layout()
        fig.savefig(out / "stage1_renders_alpha.png")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
