"""Stage 3 analysis: curves, renders, recovered-vs-GT geometry and failure analysis.

    python scripts/plot_stage3.py [--date 2026-10-11]

Per-Gaussian visibility: the rasterizer accepts any number of color channels, so rendering
the GT scene with colors = identity (N x N) yields, per pixel, each Gaussian's
compositing weight T_k alpha_k; summing over pixels and training views gives the total
"visible pixel mass" of each Gaussian.
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.camera import sphere_cameras  # noqa: E402
from src.render_torch import render  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}   # categorical slots 1-4
LABEL = {"A": "A: GT + noise", "B": "B: random, N=200", "C": "C: random, N=400",
         "D": "D: random, N=200 + relocation"}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def find_runs(date):
    runs = {"A": [], "B": [], "C": [], "D": []}
    for d in sorted((ROOT / "outputs").glob("*stage3_synthetic_*")):
        if (date and not d.name.startswith(date)) or not (d / "metrics.json").exists():
            continue
        if "gtpert" in d.name:
            runs["A"].append(d)
        elif "_reloc" in d.name:
            runs["D"].append(d)
        elif "random_n200" in d.name:
            runs["B"].append(d)
        elif "random_n400" in d.name:
            runs["C"].append(d)
    return runs


@torch.no_grad()
def visibility(run_dir: Path) -> np.ndarray:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    gt = {k: v.cuda() for k, v in torch.load(run_dir / "gt.pt").items()}
    cc = cfg.cameras
    cams = sphere_cameras(cc.n, cc.radius, cc.fov_deg, cc.width, cc.height, seed=cfg.scene.seed, device="cuda")
    train = [i for i in range(cc.n) if i % cc.test_every != 0]
    n = len(gt["means"])
    eye = torch.eye(n, device="cuda")
    vis = torch.zeros(n, device="cuda")
    for i in train:
        c = cams[i]
        img, _, _ = render(gt["means"], gt["quats"], gt["scales"], gt["opacities"], eye,
                           c.viewmat, c.K, cc.width, cc.height)
        vis += img.sum((0, 1))
    return vis.cpu().numpy() / len(train)


def gradient_stats(run_dir: Path, unrecovered: np.ndarray):
    """Per recovered Gaussian, over training views v:
        view_grad = mean_v |dL_v / d means2d|   (3DGS densification statistic, px units)
        net_grad  = |sum_v dL_v / d mu|          (what gradient descent actually follows)
    and which live Gaussians are "compromises": the nearest (Mahalanobis) live Gaussian
    whose 3-sigma ellipsoid contains an unrecovered GT Gaussian."""
    from src.gaussians import covariance_3d
    cfg = OmegaConf.load(run_dir / "config.yaml")
    gt = torch.load(run_dir / "gt.pt"); rec = torch.load(run_dir / "final.pt")
    cc = cfg.cameras
    cams = sphere_cameras(cc.n, cc.radius, cc.fov_deg, cc.width, cc.height, seed=cfg.scene.seed, device="cuda")
    train = [i for i in range(cc.n) if i % cc.test_every != 0]
    bg = torch.tensor(list(cc.bg), device="cuda")
    g = {k: v.cuda() for k, v in gt.items()}
    r = {k: v.cuda().clone() for k, v in rec.items()}
    r["means"].requires_grad_()
    n = len(r["means"])
    view_sum = torch.zeros(n, device="cuda"); view_cnt = torch.zeros(n, device="cuda")
    for i in train:
        c = cams[i]
        with torch.no_grad():
            target, _, _ = render(g["means"], g["quats"], g["scales"], g["opacities"], g["colors"],
                                  c.viewmat, c.K, cc.width, cc.height, bg=bg)
        img, _, info = render(r["means"], r["quats"], r["scales"], r["opacities"], r["colors"],
                              c.viewmat, c.K, cc.width, cc.height, bg=bg)
        info["means2d"].retain_grad()
        (img - target).abs().mean().backward()
        g2 = info["means2d"].grad.norm(dim=-1)
        vis = info["radii"] > 0
        view_sum += torch.where(vis, g2, 0.0); view_cnt += vis.float()
    net = r["means"].grad.norm(dim=-1)
    view = view_sum / view_cnt.clamp(min=1)
    alive = rec["opacities"] > 0.05
    S = covariance_3d(rec["quats"].double(), rec["scales"].double()); Si = torch.linalg.inv(S)
    comp = torch.zeros(n, dtype=torch.bool)
    for i in np.where(unrecovered)[0]:
        d = gt["means"][i].double() - rec["means"].double()
        m = torch.einsum("ni,nij,nj->n", d, Si, d); m[~alive] = 1e9
        j = int(m.argmin())
        if m[j] < 9:
            comp[j] = True
    return view.cpu().numpy(), net.cpu().numpy(), comp.numpy(), alive.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    args = ap.parse_args()
    runs = find_runs(args.date)
    OUT.mkdir(parents=True, exist_ok=True)
    M = {k: [json.loads((d / "metrics.json").read_text()) for d in v] for k, v in runs.items()}

    # ---------------- table
    print("| run | seed | N | test PSNR (dB) | recovered | median pos err | median Σ rel err | "
          "median axis err (aniso>1.5) | median naive quat err | ms/iter |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in "ABCD":
        for d, m in zip(runs[k], M[k]):
            seed = OmegaConf.load(d / "config.yaml").seed
            print(f"| {k} | {seed} | {m['n_init']} | {m['test_psnr']:.2f} | {100 * m['recovered_frac']:.1f} % | "
                  f"{m['pos_err_median']:.4f} | {m['cov_rel_err_median']:.3f} | "
                  f"{m['axis_err_deg_median_anisotropic']:.1f}° | {m['naive_quat_err_deg_median']:.1f}° | "
                  f"{m['ms_per_iter']:.0f} |")

    print("\n| group | runs | test PSNR (dB) mean ± sd | recovered mean ± sd |")
    print("|---|---|---|---|")
    for k in "ABCD":
        if M[k]:
            p = np.array([m["test_psnr"] for m in M[k]]); r = 100 * np.array([m["recovered_frac"] for m in M[k]])
            print(f"| {LABEL[k]} | {len(p)} | {p.mean():.2f} ± {p.std():.2f} | {r.mean():.1f} ± {r.std():.1f} % |")

    # ---------------- curves: test PSNR and recovered fraction vs iteration
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    for k in "ABCD":
        for j, d in enumerate(runs[k]):
            h = json.loads((d / "history.json").read_text())
            lab = LABEL[k] if j == 0 else None
            axes[0].plot([r["it"] for r in h], [r["test_psnr"] for r in h], color=SERIES[k], lw=1.6,
                         alpha=1.0 if j == 0 else 0.5, label=lab)
            hp = [r for r in h if "recovered_frac" in r]
            axes[1].plot([r["it"] for r in hp], [100 * r["recovered_frac"] for r in hp], color=SERIES[k],
                         lw=1.6, marker="o", ms=4, alpha=1.0 if j == 0 else 0.5, label=lab)
    axes[0].set_ylabel("held-out test PSNR (dB)", color=MUTED)
    axes[1].set_ylabel("GT Gaussians recovered (%)", color=MUTED)
    for ax in axes:
        ax.set_xlabel("iteration", color=MUTED); style(ax)
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "stage3_curves.png"); plt.close(fig)

    # ---------------- renders: GT | A | B | C for two test views
    first = {k: runs[k][0] for k in "ABCD" if runs[k]}
    rows = []
    for j in (0, 2):
        tiles = []
        for k, d in first.items():
            im = iio.imread(d / "renders" / f"test{j}.png")
            w = im.shape[1] // 2
            if not tiles:
                tiles.append(("ground truth", im[:, :w]))
            tiles.append((LABEL[k], im[:, w:]))
        rows.append(tiles)
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(3 * len(rows[0]), 3 * len(rows)), dpi=150)
    for r, tiles in enumerate(rows):
        for c, (t, im) in enumerate(tiles):
            axes[r, c].imshow(im, interpolation="nearest"); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(t, fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "stage3_renders.png"); plt.close(fig)

    # ---------------- geometry and failure analysis for B (seed 0)
    if runs["B"]:
        d = runs["B"][0]
        gt = torch.load(d / "gt.pt"); rec = torch.load(d / "final.pt")
        e = dict(np.load(d / "param_errors.npz"))
        ok = e["recovered"]
        vis = visibility(d)
        contrast = np.sqrt(((gt["colors"].numpy() - 1.0) ** 2).mean(-1))   # RMS distance to white bg

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        g = gt["means"].numpy(); r = rec["means"].numpy()[e["match"]]
        for i in range(len(g)):
            ax.plot([g[i, 0], r[i, 0]], [g[i, 2], r[i, 2]], color=MUTED, lw=0.6, zorder=1)
        alive = rec["opacities"].numpy() > 0.05
        ax.scatter(rec["means"][alive, 0], rec["means"][alive, 2], s=10, color="#9a9994", zorder=2,
                   label="recovered (opacity > 0.05)")
        ax.scatter(g[ok, 0], g[ok, 2], s=26, color=SERIES["A"], edgecolor="white", lw=0.8, zorder=3,
                   label=f"GT, recovered ({ok.sum()})")
        ax.scatter(g[~ok, 0], g[~ok, 2], s=30, color=SERIES["B"], marker="X", edgecolor="white", lw=0.6,
                   zorder=4, label=f"GT, not recovered ({(~ok).sum()})")
        ax.set_xlabel("x", color=MUTED); ax.set_ylabel("z", color=MUTED); ax.set_aspect("equal"); style(ax)
        ax.set_title("Run B (seed 0), top view: GT means and their Hungarian matches", fontsize=9,
                     color=INK, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        fig.tight_layout(); fig.savefig(OUT / "stage3_matches_topview.png"); plt.close(fig)

        # failure rate vs visibility and vs color contrast with the background
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)
        for ax, x, name, logx in ((axes[0], vis, "visible pixel mass per view (Σ T·α)", True),
                                  (axes[1], contrast, "RMS color distance to the white background", False)):
            ax.scatter(x[ok], e["pos_err"][ok], s=12, color=SERIES["A"], label="recovered")
            ax.scatter(x[~ok], e["pos_err"][~ok], s=16, color=SERIES["B"], marker="X", label="not recovered")
            ax.set_xlabel(name, color=MUTED); ax.set_yscale("log"); style(ax)
            if logx:
                ax.set_xscale("log")
        axes[0].set_ylabel("position error", color=MUTED)
        axes[0].legend(frameon=False, fontsize=8)
        # rotation identifiability (run B, recovered Gaussians only): principal-axis error vs
        # anisotropy, against the naive quaternion angle that ignores sign/axis equivalences
        axes[2].scatter(e["anisotropy"][ok], e["naive_quat_err_deg"][ok], s=12, color="#9a9994",
                        marker="^", label="naive quaternion angle")
        axes[2].scatter(e["anisotropy"][ok], e["axis_err_deg"][ok], s=12, color=SERIES["A"],
                        label="principal-axis error")
        axes[2].set_xscale("log"); axes[2].set_yscale("log")
        axes[2].set_xlabel("GT anisotropy s_max / s_mid", color=MUTED)
        axes[2].set_ylabel("degrees", color=MUTED); style(axes[2])
        axes[2].set_title("Run B (recovered): rotation identifiability", fontsize=9, color=INK, loc="left")
        axes[2].legend(frameon=False, fontsize=8)
        axes[0].set_title("Run B: failures vs visibility", fontsize=9, color=INK, loc="left")
        axes[1].set_title("Run B: failures vs contrast with background", fontsize=9, color=INK, loc="left")
        fig.tight_layout(); fig.savefig(OUT / "stage3_failures.png"); plt.close(fig)

        # gradient signature of compromise Gaussians
        view, net, comp, alive_r = gradient_stats(d, ~ok)
        rest = alive_r & ~comp
        fig, ax = plt.subplots(figsize=(6, 4.4), dpi=150)
        ax.scatter(view[rest], net[rest], s=12, color="#9a9994", label=f"other live Gaussians ({rest.sum()})")
        ax.scatter(view[comp], net[comp], s=22, color=SERIES["B"], marker="X", edgecolor="white", lw=0.5,
                   label=f"compromise Gaussians ({comp.sum()})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("mean per-view |∂L/∂μ₂D|  (3DGS densification statistic)", color=MUTED)
        ax.set_ylabel("net |Σ_v ∂L_v/∂μ|  (what descent follows)", color=MUTED)
        ax.set_title("Run B (seed 0), converged: densification statistic does not flag compromises", fontsize=9, color=INK, loc="left")
        style(ax); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "stage3_gradients.png"); plt.close(fig)
        med = lambda x: float(np.median(x))
        print(f"\nB s0 gradients (median): compromise view-grad {med(view[comp]):.2e} vs others {med(view[rest]):.2e} "
              f"(x{med(view[comp]) / med(view[rest]):.1f}); net-grad {med(net[comp]):.2e} vs {med(net[rest]):.2e} "
              f"(x{med(net[comp]) / med(net[rest]):.2f})")
        thr = np.quantile(view[alive_r], 0.9)
        print(f"B s0: top-10% view-grad live Gaussians contain {int((view[comp] >= thr).sum())}/{comp.sum()} compromises")

        # numbers for the write-up
        q = np.quantile(vis, 0.25)
        print(f"\nB s0: failure rate among least-visible quartile: {100 * (~ok[vis <= q]).mean():.0f} % "
              f"vs rest {100 * (~ok[vis > q]).mean():.0f} %")
        lowc = contrast < 0.2
        print(f"B s0: failure rate when color within 0.2 of bg: {100 * (~ok[lowc]).mean():.0f} % "
              f"(n={lowc.sum()}) vs rest {100 * (~ok[~lowc]).mean():.0f} %")
        dead = (rec["opacities"].numpy() < 0.05).sum()
        print(f"B s0: recovered-side Gaussians with opacity < 0.05 ('dead'): {dead}")
        far = e["pos_err"] > 0.1
        print(f"B s0: GT Gaussians with no recovered Gaussian within 0.1: {far.sum()}")

    # copy the GIF of the first B run
    if runs["B"]:
        (OUT / "stage3_progress_B.gif").write_bytes((runs["B"][0] / "progress.gif").read_bytes())
    if runs["A"]:
        (OUT / "stage3_progress_A.gif").write_bytes((runs["A"][0] / "progress.gif").read_bytes())
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
