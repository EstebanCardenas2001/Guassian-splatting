r"""Adaptive density control (ADC) of 3D Gaussians, as in 3DGS (Kerbl et al. 2023, Sec. 5.2).

The optimizer can move, reshape and fade Gaussians, but not change how many there are.
ADC does that from a single per-Gaussian statistic, the **view-space positional gradient**:

    g_k = (1 / n_k) * sum_{views v where k is visible} || dL_v / d mu'_k ||_NDC

where mu'_k is the projected 2D mean. The pixel-space gradient is converted to NDC
(normalized device coordinates, x_ndc = 2 u / W - 1) by the chain rule:
dL/dx_ndc = dL/du * du/dx_ndc = dL/du * W/2, and likewise for y with H/2. A large g_k means
the photometric loss keeps pulling the Gaussian around on screen: the region is
under-reconstructed. Every ``refine_every`` steps in [refine_start, refine_stop):

* **clone** (small Gaussians, max scale <= grow_scale3d * scene_scale, with g_k > tau):
  under-reconstruction - add an identical copy; the two then separate along the gradient.
* **split** (large Gaussians with g_k > tau): over-reconstruction - one Gaussian covers
  detail it can't represent; replace it by 2 children sampled from it,
  mu_child = mu + R diag(s) eps, eps ~ N(0, I), with scales s / 1.6.
* **prune** Gaussians with opacity < prune_opa, and (after the first reset period) those
  whose world-space max scale exceeds prune_scale3d * scene_scale.
* **opacity reset** every ``reset_every`` steps: o <- min(o, 2 prune_opa). Floaters and
  redundant Gaussians that don't get their opacity back are pruned at the next refinement.

New Gaussians start with zero Adam moments; removed ones lose theirs. Parameters are
replaced in the ``ParameterDict`` and each tensor's optimizer state is rewritten to match.

Interface and defaults follow gsplat's ``DefaultStrategy`` (1.5.3), so the two can be
swapped in ``src/train.py`` and compared. One deliberate difference: gsplat 1.5.3 never
performs the opacity reset (its condition ``step % reset_every == 0 & step > 0`` is a chained
comparison that is always False). Here ``opacity_reset=True`` does what 3DGS describes, and
``opacity_reset=False`` reproduces gsplat's actual behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.gaussians import quat_to_rotmat


@dataclass
class ADC:
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002          # tau, on the NDC-space gradient norm
    grow_scale3d: float = 0.01           # clone vs split threshold, x scene_scale
    prune_scale3d: float = 0.1           # "too big" in world space, x scene_scale
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    reset_every: int = 3000
    opacity_reset: bool = True
    absgrad: bool = False                # AbsGS (Stage 8): accumulate sum |dL/dpixel| instead
    verbose: bool = False

    # ------------------------------------------------------------------ interface
    def initialize_state(self, scene_scale: float = 1.0) -> dict:
        return {"grad2d": None, "count": None, "scene_scale": scene_scale, "log": []}

    def check_sanity(self, params, optimizers):
        for k in ("means", "scales", "quats", "opacities"):
            assert k in params, f"missing parameter {k}"
        for k, p in params.items():
            assert k in optimizers and optimizers[k].param_groups[0]["params"][0] is p, k

    def step_pre_backward(self, params, optimizers, state, step, info):
        key = "means2d"
        info[key].retain_grad()          # non-leaf: keep its gradient for the statistic

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, state, step, info, packed: bool = True):
        if step >= self.refine_stop_iter:
            return
        self._accumulate(params, state, info, packed)
        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_clone, n_split = self._grow(params, optimizers, state)
            n_prune = self._prune(params, optimizers, state, step)
            state["grad2d"].zero_()
            state["count"].zero_()
            state["log"].append((step, n_clone, n_split, n_prune, len(params["means"])))
            if self.verbose:
                print(f"step {step}: +{n_clone} cloned, {n_split} split, -{n_prune} pruned "
                      f"-> {len(params['means'])}")
        if self.opacity_reset and step > 0 and step % self.reset_every == 0:
            cap = torch.logit(torch.tensor(2 * self.prune_opa)).item()
            _rewrite(params, optimizers, state,
                     param_fn=lambda k, p: p.clamp(max=cap) if k == "opacities" else p,
                     moment_fn=lambda k, v: torch.zeros_like(v) if k == "opacities" else v,
                     names=["opacities"])

    # ------------------------------------------------------------------ statistic
    def _accumulate(self, params, state, info, packed):
        g = info["means2d"].absgrad if self.absgrad else info["means2d"].grad
        g = g.clone()
        g[..., 0] *= info["width"] / 2.0 * info["n_cameras"]       # pixels -> NDC
        g[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(params["means"])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=g.device)
            state["count"] = torch.zeros(n, device=g.device)
        if packed:
            ids = info["gaussian_ids"]                                 # visible Gaussians (nnz,)
        else:
            vis = (info["radii"] > 0).all(-1)                           # (C, N)
            ids = torch.where(vis)[1]
            g = g[vis]
        state["grad2d"].index_add_(0, ids, g.norm(dim=-1))
        state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    # ------------------------------------------------------------------ clone / split
    def _grow(self, params, optimizers, state) -> tuple[int, int]:
        grad = state["grad2d"] / state["count"].clamp_min(1)
        high = grad > self.grow_grad2d
        small = torch.exp(params["scales"]).amax(-1) <= self.grow_scale3d * state["scene_scale"]
        clone, split = high & small, high & ~small
        n_clone, n_split = int(clone.sum()), int(split.sum())

        if n_clone:
            sel = torch.where(clone)[0]
            _rewrite(params, optimizers, state,
                     param_fn=lambda k, p: torch.cat([p, p[sel]]),
                     moment_fn=lambda k, v: torch.cat([v, v.new_zeros((len(sel), *v.shape[1:]))]),
                     state_fn=lambda v: torch.cat([v, v[sel]]))
            split = torch.cat([split, split.new_zeros(n_clone)])       # clones are never split

        if n_split:
            sel, rest = torch.where(split)[0], torch.where(~split)[0]
            s = torch.exp(params["scales"][sel])                       # (S, 3)
            R = quat_to_rotmat(params["quats"][sel])                   # (S, 3, 3), normalizes
            eps = torch.randn(2, len(sel), 3, device=s.device)         # same draw order as gsplat
            offsets = torch.einsum("nij,nj,bnj->bni", R, s, eps)       # R diag(s) eps, (2, S, 3)

            def param_fn(k, p):
                if k == "means":
                    child = (p[sel] + offsets).reshape(-1, 3)
                elif k == "scales":
                    child = torch.log(s / 1.6).repeat(2, 1)
                else:
                    child = p[sel].repeat(2, *[1] * (p.dim() - 1))
                return torch.cat([p[rest], child])

            _rewrite(params, optimizers, state, param_fn=param_fn,
                     moment_fn=lambda k, v: torch.cat([v[rest], v.new_zeros((2 * len(sel), *v.shape[1:]))]),
                     state_fn=lambda v: torch.cat([v[rest], v[sel].repeat(2)]))
        return n_clone, n_split

    # ------------------------------------------------------------------ prune
    def _prune(self, params, optimizers, state, step) -> int:
        prune = torch.sigmoid(params["opacities"]) < self.prune_opa
        if step > self.reset_every:
            prune |= torch.exp(params["scales"]).amax(-1) > self.prune_scale3d * state["scene_scale"]
        n = int(prune.sum())
        if n:
            keep = torch.where(~prune)[0]
            _rewrite(params, optimizers, state, param_fn=lambda k, p: p[keep],
                     moment_fn=lambda k, v: v[keep], state_fn=lambda v: v[keep])
        return n


def _rewrite(params, optimizers, state, param_fn, moment_fn, state_fn=None, names=None):
    """Replace parameters ``params[k] <- param_fn(k, params[k])`` and keep each optimizer
    consistent: its per-element state tensors (Adam's exp_avg, exp_avg_sq) are mapped with
    ``moment_fn`` and re-keyed to the new Parameter; the scalar ``step`` is kept."""
    for k in names or list(params.keys()):
        old = params[k]
        new = torch.nn.Parameter(param_fn(k, old.data), requires_grad=old.requires_grad)
        params[k] = new
        opt = optimizers[k]
        st = opt.state.pop(old, {})
        for key, v in st.items():
            if key != "step":
                st[key] = moment_fn(k, v)
        opt.param_groups[0]["params"] = [new]
        opt.state[new] = st
    if state_fn is not None:
        for key in ("grad2d", "count"):
            if isinstance(state.get(key), Tensor):
                state[key] = state_fn(state[key])
