"""Diagnostic: run my ADC inside a real training loop and, at every densification step,
apply gsplat's DefaultStrategy to a clone of the full state (params, Adam moments, statistics)
with the same RNG seed. Report the first step where the two disagree.

    python -m scripts.adc_lockstep [iterations] [downscale]
"""
import copy
import sys

import torch
from gsplat.strategy import DefaultStrategy

from src.config import load_config
from src.densify import ADC
from src.losses import photometric_loss
from src.run import REPO_ROOT
from src.train import LR_KEYS, get_renderer, init_model, load_data


def clone_all(params, opts, state):
    p2 = torch.nn.ParameterDict({k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()})
    o2 = {}
    for k, o in opts.items():
        o2[k] = torch.optim.Adam([p2[k]], lr=o.param_groups[0]["lr"], eps=1e-15)
        o2[k].state[p2[k]] = {kk: vv.clone() for kk, vv in o.state[params[k]].items()}
    s2 = {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in state.items()
          if k != "log"}
    s2.setdefault("radii", None)
    return p2, o2, s2


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    ds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cfg = load_config(REPO_ROOT / "configs/stage5_lego_adc.yaml", [f"data.downscale={ds}"])
    torch.manual_seed(0)
    dev = torch.device("cuda")
    cams, imgs, _, _ = load_data(cfg, dev)
    bg = torch.ones(3, device=dev)
    render_fn = get_renderer(cfg)
    model = init_model(cfg, dev, torch.Generator(device=dev).manual_seed(0))
    opts = {key: torch.optim.Adam([model.params[key]], lr=cfg.train.lr[name], eps=1e-15)
            for name, key in LR_KEYS.items()}
    centers = torch.stack([c.center for c in cams])
    scene_scale = float(1.1 * (centers - centers.mean(0)).norm(dim=-1).max())
    mine, ref = ADC(opacity_reset=False), DefaultStrategy()
    state = mine.initialize_state(scene_scale)
    for it in range(1, iters + 1):
        i = it % len(cams)
        img, _, info = render_fn(model, cams[i], bg)
        mine.step_pre_backward(model.params, opts, state, it, info["meta"])
        photometric_loss(img, imgs[i].float() / 255).backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        refine = it > 500 and it % 100 == 0 and it < 15000
        if refine:
            p2, o2, s2 = clone_all(model.params, opts, state)
            torch.manual_seed(it); ref.step_post_backward(p2, o2, s2, it, info["meta"], packed=True)
        torch.manual_seed(it); mine.step_post_backward(model.params, opts, state, it, info["meta"], packed=True)
        if refine:
            same = all(model.params[k].shape == p2[k].shape and torch.equal(model.params[k], p2[k])
                       for k in model.params)
            same_m = all(torch.equal(opts[k].state[model.params[k]][m], o2[k].state[p2[k]][m])
                         for k in model.params for m in ("exp_avg", "exp_avg_sq")) if same else False
            if not (same and same_m):
                print(f"MISMATCH at step {it}: mine N={len(model.params['means'])}, gsplat N={len(p2['means'])}, "
                      f"params equal={same}, moments equal={same_m}")
                for k in model.params:
                    print("  ", k, tuple(model.params[k].shape), tuple(p2[k].shape))
                return
            if it % 1000 == 0:
                print(f"step {it}: identical, N={len(model.params['means'])}", flush=True)
    print("no mismatch")


if __name__ == "__main__":
    main()
