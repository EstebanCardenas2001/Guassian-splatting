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
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage4", required=True)
    args = ap.parse_args()
    run, run4 = Path(args.run), Path(args.stage4)

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

    # ---- renders: GT | Stage 4 (400², upsampled) | Stage 5 (800²), full + zoom
    f5 = sorted((run / "renders").glob("test_*.png"))[:2]
    fig, axes = plt.subplots(2 * len(f5), 3, figsize=(10.5, 7 * len(f5)), dpi=150)
    for k, f in enumerate(f5):
        im5 = iio.imread(f); w = im5.shape[1] // 2
        gt, r5 = im5[:, :w], im5[:, w:]
        im4 = iio.imread(run4 / "renders" / f.name); w4 = im4.shape[1] // 2
        r4 = np.repeat(np.repeat(im4[:, w4:], 2, 0), 2, 1)          # 400 -> 800, nearest
        y0, x0, s = 330, 300, 160                                    # zoom on the bucket / tracks
        for c, (im, t) in enumerate(((gt, "ground truth 800²"), (r4, "Stage 4: PyTorch, 400² (×2)"),
                                     (r5, "Stage 5: gsplat, 800²"))):
            axes[2 * k, c].imshow(im); axes[2 * k + 1, c].imshow(im[y0:y0 + s, x0:x0 + s], interpolation="nearest")
            if k == 0:
                axes[0, c].set_title(t, fontsize=9, color=INK)
            for a in (axes[2 * k, c], axes[2 * k + 1, c]):
                a.set_xticks([]); a.set_yticks([])
            axes[2 * k, c].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="#eb6834", lw=1))
    fig.tight_layout(); fig.savefig(OUT / "stage5_renders.png"); plt.close(fig)

    # ---- test PSNR curve, both stages
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    for r, b, lab in ((run4, "torch", "Stage 4: PyTorch, 400², 100k"), (run, "gsplat", "Stage 5: gsplat, 800², 1M")):
        ea = EventAccumulator(str(r / "tb")); ea.Reload()
        ev = ea.Scalars("test/psnr")
        ax.plot([e.step for e in ev], [e.value for e in ev], color=BACKEND[b][0], lw=2, marker="o", ms=3, label=lab)
    ax.axhline(32, color=MUTED, lw=1, ls="--")
    ax.annotate("32 dB target", (0, 32), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("iteration", color=MUTED); ax.set_ylabel("test PSNR (dB, 20 views)", color=MUTED); style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "stage5_curves.png"); plt.close(fig)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
