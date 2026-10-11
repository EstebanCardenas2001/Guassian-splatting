"""Measured agreement between render_torch (with gsplat's constants) and render_gsplat.

    python -m scripts.stage5_agreement
Prints image / alpha differences and per-parameter gradient agreement on random scenes,
and writes docs/figures/stage5_agreement.png (both renders + |difference|).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.render_gsplat import GSPLAT_ALPHA_MAX, GSPLAT_Q_CUTOFF  # noqa: E402
from src.render_gsplat import render as render_gs  # noqa: E402
from src.render_torch import render as render_pt  # noqa: E402
from tests.test_render_gsplat import camera, scene  # noqa: E402

DEV = "cuda"
NAMES = ("means", "quats", "scales", "opacities", "colors")


def main():
    bg = torch.tensor([1.0, 1.0, 1.0], device=DEV)
    print("| scene | mode | max abs Δ image | mean abs Δ image | max abs Δ alpha |")
    print("|---|---|---|---|---|")
    for seed in (0, 1, 2):
        for aa in (False, True):
            p = scene(N=300, seed=seed)
            cam = camera(W=160, H=120, f=150.0)
            a, aa_alpha, _ = render_pt(*p, *cam, bg=bg, antialiased=aa, alpha_max=GSPLAT_ALPHA_MAX, q_cutoff=GSPLAT_Q_CUTOFF)
            b, bb_alpha, _ = render_gs(*p, *cam, bg=bg, antialiased=aa)
            d = (a - b).abs()
            print(f"| seed {seed} | {'antialiased' if aa else 'classic'} | {d.max():.2e} | {d.mean():.2e} | "
                  f"{(aa_alpha - bb_alpha).abs().max():.2e} |")
            if seed == 0 and not aa:
                img_pair = (a, b, d)
    print("\n| parameter | cosine(∇torch, ∇gsplat) | ‖Δ∇‖ / ‖∇torch‖ |")
    print("|---|---|---|")
    base = scene(N=300, seed=5)
    cam = camera(W=160, H=120, f=150.0)
    w = torch.randn(120, 160, 3, generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
    grads = []
    for fn, kw in ((render_pt, dict(alpha_max=GSPLAT_ALPHA_MAX, q_cutoff=GSPLAT_Q_CUTOFF)), (render_gs, {})):
        p = [t.clone().requires_grad_() for t in base]
        (fn(*p, *cam, bg=bg, **kw)[0] * w).sum().backward()
        grads.append([t.grad for t in p])
    for n, ga, gb in zip(NAMES, *grads):
        cos = torch.nn.functional.cosine_similarity(ga.flatten(), gb.flatten(), dim=0)
        print(f"| {n} | {cos:.6f} | {(ga - gb).norm() / ga.norm():.2e} |")

    a, b, d = img_pair
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=150)
    for ax, im, t in zip(axes, (a, b), ("render_torch (gsplat constants)", "render_gsplat")):
        ax.imshow(im.clamp(0, 1).cpu()); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    m = axes[2].imshow(d.amax(-1).cpu(), cmap="magma"); axes[2].set_title("max-channel |difference|", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([]); fig.colorbar(m, ax=axes[2], fraction=0.035)
    fig.tight_layout(); fig.savefig("docs/figures/stage5_agreement.png")


if __name__ == "__main__":
    main()
