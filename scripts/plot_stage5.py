"""Stage 5 figures: backend benchmark, 800x800 lego renders with zooms vs Stage 4, curves.

    python -m scripts.plot_stage5 --run outputs/<stage5 run> --stage4 outputs/<stage4 run>
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
BACKEND = {"torch": ("#eb6834", "pure PyTorch (Stage 2)"), "gsplat": ("#2a78d6", "gsplat (tiled CUDA)")}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="best Stage 5 run (densified)")
    ap.add_argument("--fixed", required=True, help="best Stage 5 run without densification")
    ap.add_argument("--stage4", required=True)
    args = ap.parse_args()
    run, runf, run4 = Path(args.run), Path(args.fixed), Path(args.stage4)

    # ---- benchmark: ms and GB vs N, one panel per resolution
    prof = {b: json.load(open(f"/mnt/outputs/profiles/{b}_trained.json")) for b in ("torch", "gsplat")}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=150, sharex=True)
    for j, res in enumerate((200, 400, 800)):
        for b, rows in prof.items():
            col, lab = BACKEND[b]
            ok = [r for r in rows if r["res"] == res and r["status"] == "ok"]
            oom = [r for r in rows if r["res"] == res and r["status"] != "ok"]
            for i, key in enumerate(("ms", "peak_GB")):
                axes[i, j].plot([r["n"] for r in ok], [r[key] for r in ok], color=col, lw=2, marker="o",
                                ms=5, label=lab)
                for r in oom:            # mark out-of-memory configurations at the top edge
                    axes[i, j].annotate("OOM", (r["n"], 1.0), color=col, fontsize=8, ha="center",
                                        xycoords=("data", "axes fraction"), xytext=(0, -12),
                                        textcoords="offset points")
        axes[0, j].set_title(f"{res}×{res}", fontsize=10, color=INK, loc="left")
        for i in range(2):
            axes[i, j].set_xscale("log"); axes[i, j].set_yscale("log"); style(axes[i, j])
        axes[1, j].set_xlabel("Gaussians N (trained lego, resampled)", color=MUTED)
    axes[0, 0].set_ylabel("ms per forward + backward", color=MUTED)
    axes[1, 0].set_ylabel("peak VRAM (GB)", color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "stage5_benchmark.png"); plt.close(fig)

    # ---- renders: GT | Stage 4 (400², upsampled) | Stage 5 fixed set | Stage 5 densified, full + zoom
    f5 = sorted((run / "renders").glob("test_*.png"))[:2]
    cols = 4
    fig, axes = plt.subplots(2 * len(f5), cols, figsize=(3.4 * cols, 7 * len(f5)), dpi=150)
    for k, f in enumerate(f5):
        def half(p, right=True):
            im = iio.imread(p); w = im.shape[1] // 2
            return im[:, w:] if right else im[:, :w]
        gt, r5 = half(f, False), half(f)
        rf = half(runf / "renders" / f.name)
        r4 = np.repeat(np.repeat(half(run4 / "renders" / f.name), 2, 0), 2, 1)   # 400 -> 800, nearest
        y0, x0, s = 330, 300, 160                                    # zoom window
        tiles = ((gt, "ground truth 800²"), (r4, "Stage 4: PyTorch 400², 100k fixed (×2)"),
                 (rf, "Stage 5: gsplat 800², 300k fixed"), (r5, "Stage 5: gsplat 800² + densification"))
        for c, (im, t) in enumerate(tiles):
            axes[2 * k, c].imshow(im); axes[2 * k + 1, c].imshow(im[y0:y0 + s, x0:x0 + s], interpolation="nearest")
            if k == 0:
                axes[0, c].set_title(t, fontsize=8, color=INK)
            for a in (axes[2 * k, c], axes[2 * k + 1, c]):
                a.set_xticks([]); a.set_yticks([])
            axes[2 * k, c].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="#eb6834", lw=1))
    fig.tight_layout(); fig.savefig(OUT / "stage5_renders.png"); plt.close(fig)

    # ---- test PSNR curve, both stages
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    runs = ((run4, "#9a9994", "Stage 4: PyTorch 400², 100k fixed"),
            (runf, BACKEND["torch"][0], "Stage 5: gsplat 800², 300k fixed"),
            (run, BACKEND["gsplat"][0], "Stage 5: gsplat 800² + densification"))
    for r, col, lab in runs:
        ea = EventAccumulator(str(r / "tb")); ea.Reload()
        ev = ea.Scalars("test/psnr")
        ax.plot([e.step for e in ev], [e.value for e in ev], color=col, lw=2, marker="o", ms=3, label=lab)
    ax.axhline(32, color=MUTED, lw=1, ls="--")
    ax.annotate("32 dB target", (0, 32), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("iteration", color=MUTED); ax.set_ylabel("test PSNR (dB, 20 views)", color=MUTED); style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "stage5_curves.png"); plt.close(fig)
    print(f"figures -> {OUT}")


if __name__ == "__main__" and "--densify-runs" not in __import__("sys").argv:
    main()


def plot_densify_runs():
    """Gaussian count and test PSNR over training for every densified lego run."""
    groups = {"gsplat DefaultStrategy (no reset)": ("#9a9994", ["stage5_lego_gsplat_densify", "stage5_lego_gsplat_densify_rep"]),
              "my ADC, reset off": ("#eb6834", ["stage5_lego_adc_noreset", "stage5_lego_adc_noreset_rep"]),
              "my ADC, reset on (3DGS)": ("#2a78d6", ["stage5_lego_adc", "stage5_lego_adc_rep"])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
    for lab, (col, names) in groups.items():
        for j, n in enumerate(names):
            ea = EventAccumulator(str(ROOT / "outputs" / f"2026-10-11_{n}" / "tb"), size_guidance={"scalars": 0}); ea.Reload()
            g = ea.Scalars("model/n_gaussians"); p = ea.Scalars("test/psnr")
            kw = dict(color=col, lw=1.8, ls="-" if j == 0 else "--", label=lab if j == 0 else None)
            axes[0].plot([e.step for e in g], [e.value / 1e3 for e in g], **kw)
            axes[1].plot([e.step for e in p], [e.value for e in p], **kw)
    for x in range(3000, 15001, 3000):
        axes[0].axvline(x, color=GRID, lw=0.8, zorder=0)
    axes[0].set_ylabel("Gaussians (thousands)", color=MUTED); axes[1].set_ylabel("test PSNR (dB, 20 views)", color=MUTED)
    axes[1].set_ylim(26, 34.8)
    for ax in axes:
        ax.set_xlabel("iteration", color=MUTED); style(ax)
    axes[0].legend(frameon=False, fontsize=8, title="solid / dashed = two runs", title_fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "stage5_densify_runs.png"); plt.close(fig)


if __name__ == "__main__" and "--densify-runs" in __import__("sys").argv:
    plot_densify_runs()
