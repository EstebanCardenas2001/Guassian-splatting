"""Stage 7 analysis: ablation table, test renders with zoomed crops, training curves.

    python -m scripts.plot_stage7 [--date 2026-10-12]
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
RUNS = {"A": ("baseline", "#9a9994"), "B": ("+ mask loss", "#eda100"), "C": ("+ densification", "#1baf7a"),
        "D": ("+ SH (view-dependent)", "#2a78d6"), "E": ("D, random init", "#eb6834")}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def find(date):
    out = {}
    for k in RUNS:
        cands = sorted(d for d in (ROOT / "outputs").glob(f"*stage7_{k}_*") if (d / "metrics.json").exists()
                       and (not date or d.name.startswith(date)))
        if cands:
            out[k] = cands[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--view", type=int, default=2, help="index of the saved test render to show")
    args = ap.parse_args()
    runs = find(args.date)
    M = {k: json.loads((d / "metrics.json").read_text()) for k, d in runs.items()}

    print("| run | change | PSNR | PSNR (plant only) | SSIM | LPIPS ↓ | alpha IoU | Gaussians | time |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, m in M.items():
        print(f"| {k} | {RUNS[k][0]} | {m['test_psnr']:.2f} | {m['test_psnr_masked']:.2f} | {m['test_ssim']:.4f} | "
              f"{m['test_lpips']:.4f} | {m['test_alpha_iou']:.4f} | {m['n_gaussians'] / 1e3:.0f}k | "
              f"{m['train_time_s'] / 60:.0f} min |")

    # ---- curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), dpi=150)
    for k, d in runs.items():
        ea = EventAccumulator(str(d / "tb"), size_guidance={"scalars": 0}); ea.Reload()
        for ax, tag in zip(axes, ("test/psnr_masked", "test/lpips", "model/n_gaussians")):
            ev = ea.Scalars(tag)
            y = [e.value / (1e3 if "n_gauss" in tag else 1) for e in ev]
            ax.plot([e.step for e in ev], y, color=RUNS[k][1], lw=1.8, label=f"{k}: {RUNS[k][0]}")
    for ax, lab in zip(axes, ("test PSNR inside the mask (dB)", "test LPIPS (lower is better)", "Gaussians (thousands)")):
        ax.set_ylabel(lab, color=MUTED); ax.set_xlabel("iteration", color=MUTED); style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "stage7_curves.png"); plt.close(fig)

    # ---- renders: GT + each run, full view and two zooms (largest-detail windows chosen on GT)
    files = {k: sorted((d / "renders").glob("test_*.png")) for k, d in runs.items()}
    k0 = next(iter(files))
    f = files[k0][args.view]
    im = iio.imread(f); w = im.shape[1] // 2
    gt = im[:, :w]
    H, W = gt.shape[:2]
    # zoom windows: the two 200x200 GT windows with the most gradient energy (edges, stems)
    g = np.abs(np.diff(gt.mean(-1), axis=0))[:, :-1] + np.abs(np.diff(gt.mean(-1), axis=1))[:-1]
    s = 200
    scores = [(g[y:y + s, x:x + s].sum(), y, x) for y in range(0, H - s, 50) for x in range(0, W - s, 50)]
    scores.sort(reverse=True)
    wins = [scores[0][1:]]
    for _, y, x in scores[1:]:
        if all(abs(y - yy) > s or abs(x - xx) > s for yy, xx in wins):
            wins.append((y, x)); break
    cols = [("ground truth", gt)] + [(f"{k}: {RUNS[k][0]}", iio.imread(files[k][args.view])[:, w:]) for k in runs]
    fig, axes = plt.subplots(3, len(cols), figsize=(2.9 * len(cols), 10), dpi=150,
                             gridspec_kw={"height_ratios": [1.6, 1, 1]})
    for c, (t, img) in enumerate(cols):
        axes[0, c].imshow(img); axes[0, c].set_title(t, fontsize=8, color=INK)
        for (y, x), col in zip(wins, ("#eb6834", "#2a78d6")):
            axes[0, c].add_patch(plt.Rectangle((x, y), s, s, fill=False, ec=col, lw=1))
        for r, (y, x) in enumerate(wins, 1):
            axes[r, c].imshow(img[y:y + s, x:x + s], interpolation="nearest")
        for a in axes[:, c]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "stage7_renders.png"); plt.close(fig)
    print(f"figures -> {OUT}  (view {f.name})")


if __name__ == "__main__":
    main()
