"""Stage 9: final ablation table (A-E + best Stage 8, all at full resolution) and
side-by-side crops of leaf edges and stems.

    python -m scripts.plot_stage9
Crop windows are chosen from the ground-truth mask, not by eye:
  * stem window: most medial-axis pixels of *thin* foreground (distance to the boundary
    between 2 and 12 px at full resolution: stems and petioles), at least 20 % foreground;
  * leaf-edge window: most boundary pixels of leaf *blades* (foreground > 60 px deep within
    ~60 px), away from the stem window.
"""
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage.morphology import medial_axis  # noqa: E402

from src.camera import load_colmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
INK, MUTED = "#0b0b0b", "#52514e"
RUNS = [("A: baseline", "stage7_A_baseline"), ("B: + mask loss", "stage7_B_mask"),
        ("C: + densification", "stage7_C_densify"), ("D: + spherical harmonics", "stage7_D_sh"),
        ("E: D, random init", "stage7_E_randinit"), ("best Stage 8: 2000 px × 60k", "stage8_adc_2000_60k")]


def run_dir(tag):
    return sorted((ROOT / "outputs").glob(f"*{tag}"))[-1]


def full_metrics(d):
    f = d / "metrics_full.json"
    if f.exists():
        return json.loads(f.read_text())
    m = json.loads((d / "metrics.json").read_text())
    return {k[5:]: v for k, v in m.items() if k.startswith("test_")}


def render_of(d, i):
    for sub in ("renders_full", "renders"):
        f = d / sub / f"test_{i:03d}.png"
        if f.exists():
            im = iio.imread(f)
            if im.shape[0] == 4032:
                return im[:, im.shape[1] // 2:]
    raise FileNotFoundError(d, i)


def windows(mask, s=300):
    fg = mask > 127
    dist = ndimage.distance_transform_edt(fg)
    skel = medial_axis(fg)
    thin = skel & (dist >= 2) & (dist <= 12)
    # boundary pixels of leaf blades: within ~60 px inward the foreground gets > 60 px deep
    # (blades are hundreds of px wide at full res; petioles and stems only 20-40 px)
    edge = fg & ~ndimage.binary_erosion(fg, iterations=2) & (ndimage.maximum_filter(dist, 121) > 60)
    H, W = mask.shape
    fg_frac = ndimage.uniform_filter(fg.astype(np.float32), s)

    def best(score_map, avoid=None, min_fg=0.0):
        integ = ndimage.uniform_filter(score_map.astype(np.float32), s) * s * s
        integ[fg_frac < min_fg] = 0                       # skip nearly empty windows
        integ[: s // 2] = integ[-s // 2:] = 0
        integ[:, : s // 2] = integ[:, -s // 2:] = 0
        if avoid is not None:
            ay, ax = avoid
            integ[max(0, ay - s): ay + s, max(0, ax - s): ax + s] = 0
        y, x = np.unravel_index(np.argmax(integ), integ.shape)
        return int(y), int(x)
    stem = best(thin, min_fg=0.2)
    leaf = best(edge, avoid=stem, min_fg=0.2)
    return [("thin parts", stem), ("blade edges", leaf)], s


def main():
    dirs = [(name, run_dir(tag)) for name, tag in RUNS]
    print("| run | PSNR | PSNR (plant only) | SSIM | LPIPS ↓ | Gaussians |")
    print("|---|---|---|---|---|---|")
    for name, d in dirs:
        m = full_metrics(d)
        n = json.loads((d / "metrics.json").read_text())["n_gaussians"]
        print(f"| {name} | {m['psnr']:.2f} | {m['psnr_masked']:.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {n / 1e3:.0f}k |")

    e = load_colmap(ROOT / "data/plant", res="full", load_split="test")
    show = [dirs[0], dirs[3], dirs[5]]                 # A, D, best Stage 8
    views = (1, 2, 3)
    rows = []
    for i in views:
        gt_img = e["test_images"][i].numpy()
        mask = e["test_masks"][i].numpy()
        m = mask[..., None].astype(np.float32) / 255
        gt = (m * gt_img + (1 - m) * 255).astype(np.uint8)
        wins, s = windows(mask)
        for label, (y, x) in wins:
            y0, x0 = y - s // 2, x - s // 2
            tiles = [("ground truth", gt[y0:y0 + s, x0:x0 + s])]
            tiles += [(name, render_of(d, i)[y0:y0 + s, x0:x0 + s]) for name, d in show]
            rows.append((f"view {i} · {label}", tiles))
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.75 * len(rows)), dpi=150)
    for r, (rl, tiles) in enumerate(rows):
        for c, (t, img) in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(t, fontsize=9, color=INK)
        axes[r, 0].set_ylabel(rl, fontsize=8, color=MUTED)
    fig.suptitle("Held-out views at full resolution (300×300 px crops of 3024×4032), windows chosen from the mask",
                 fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "stage9_edges_stems.png"); plt.close(fig)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
