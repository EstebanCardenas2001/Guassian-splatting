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


# =============================================================================== MCMC
def relocation_table(n_max: int = 51) -> Tensor:
    """A[n, k] = sum_{i=k+1}^{n} C(i-1, k) (-1)^k / sqrt(k+1), for n, k < n_max.

    With it the 3DGS-MCMC scale denominator (Kheradmand et al. 2024, Eq. 9)
        sum_{i=1}^{n} sum_{k=0}^{i-1} C(i-1, k) (-1)^k o'^(k+1) / sqrt(k+1)
    becomes the polynomial sum_k A[n, k] o'^(k+1), evaluated for all Gaussians at once.
    """
    import math
    A = torch.zeros(n_max + 1, n_max, dtype=torch.float64)
    for n in range(1, n_max + 1):
        for k in range(n):
            A[n, k] = sum(math.comb(i - 1, k) for i in range(k + 1, n + 1)) * (-1) ** k / math.sqrt(k + 1)
    return A


def relocate_opacity_scale(o: Tensor, s: Tensor, n: Tensor, A: Tensor) -> tuple[Tensor, Tensor]:
    """Opacity and scale for each of n copies of a Gaussian (o (M,), s (M, 3), n (M,) >= 1).

    o' = 1 - (1 - o)^(1/n): n copies stacked at the same point block the same fraction of light
    as the original at its center, 1 - (1 - o')^n = o. The scale factor o / sum_k A[n,k] o'^(k+1)
    keeps the integrated opacity profile of the n copies equal to the original's (paper, Eq. 9).
    """
    n = n.clamp(1, A.shape[0] - 1)
    o_new = 1 - (1 - o) ** (1.0 / n)
    powers = o_new[:, None] ** torch.arange(1, A.shape[1] + 1, device=o.device, dtype=o.dtype)
    denom = (A.to(o.device, o.dtype)[n.long()] * powers).sum(-1)
    return o_new, s * (o / denom)[:, None]


@dataclass
class MCMC:
    """3DGS as Markov Chain Monte Carlo (Kheradmand et al. 2024), with a hard cap on N.

    Training is read as Stochastic Gradient Langevin Dynamics: Gaussians are samples of a
    distribution proportional to how well they explain the images. So instead of clone/split
    heuristics:
      * every ``refine_every`` steps, **relocate** dead Gaussians (opacity <= min_opacity) onto
        live ones sampled with probability proportional to opacity, correcting opacity and
        scale (``relocate_opacity_scale``) so the rendered image doesn't jump, and **add**
        ``grow_factor`` x N new ones the same way until ``cap_max``;
      * every step, **inject noise** into the means, x += lr * noise_lr * g(o) * Sigma * eps,
        eps ~ N(0, I), with g(o) = sigmoid(100 (1 - o - 0.995)) ~ 1 only for nearly
        transparent Gaussians: they explore, opaque ones stay put.
    The training loss adds L1 penalties on opacity and scale (train.opacity_reg/scale_reg),
    which keep Gaussians small and let unneeded ones die and be recycled.

    Same rules as gsplat's MCMCStrategy, in PyTorch. One deliberate difference: gsplat
    zeroes the Adam moments of the sampled targets only; here the overwritten dead slots are
    zeroed too (they hold a new Gaussian, so their old moments are meaningless).
    """
    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100
    min_opacity: float = 0.005
    grow_factor: float = 0.05
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> dict:
        return {"A": relocation_table(), "log": [], "scene_scale": scene_scale}

    def check_sanity(self, params, optimizers):
        ADC.check_sanity(self, params, optimizers)

    def step_pre_backward(self, params, optimizers, state, step, info):
        pass                                   # no gradient statistics needed

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, state, step, info, packed: bool = True,
                           lr: float = 0.0):
        if self.refine_start_iter < step < self.refine_stop_iter and step % self.refine_every == 0:
            n_reloc = self._relocate(params, optimizers, state)
            n_add = self._add(params, optimizers, state)
            state["log"].append((step, n_reloc, n_add, len(params["means"])))
            if self.verbose:
                print(f"step {step}: relocated {n_reloc}, added {n_add} -> {len(params['means'])}")
        self._noise(params, lr)

    def _sample(self, probs: Tensor, n: int) -> Tensor:
        return torch.multinomial(probs, n, replacement=True)

    def _new_values(self, params, state, idx):
        o = torch.sigmoid(params["opacities"])
        copies = torch.bincount(idx)[idx] + 1                    # original + the new copies
        o_new, s_new = relocate_opacity_scale(o[idx], torch.exp(params["scales"])[idx], copies, state["A"])
        eps = torch.finfo(torch.float32).eps
        return torch.logit(o_new.clamp(self.min_opacity, 1 - eps)), torch.log(s_new)

    def _relocate(self, params, optimizers, state) -> int:
        o = torch.sigmoid(params["opacities"])
        dead = o <= self.min_opacity
        n = int(dead.sum())
        if n == 0 or n == len(o):
            return 0
        dead_idx, alive_idx = torch.where(dead)[0], torch.where(~dead)[0]
        src = alive_idx[self._sample(o[alive_idx], n)]
        op_new, sc_new = self._new_values(params, state, src)

        def param_fn(k, p):
            p = p.clone()
            if k == "opacities":
                p[src] = op_new
            elif k == "scales":
                p[src] = sc_new
            p[dead_idx] = p[src]
            return p

        def moment_fn(k, v):
            v = v.clone()
            v[src] = 0
            v[dead_idx] = 0
            return v
        _rewrite(params, optimizers, state, param_fn, moment_fn)
        return n

    def _add(self, params, optimizers, state) -> int:
        N = len(params["means"])
        n = max(0, min(self.cap_max, int((1 + self.grow_factor) * N)) - N)
        if n == 0:
            return 0
        src = self._sample(torch.sigmoid(params["opacities"]), n)
        op_new, sc_new = self._new_values(params, state, src)

        def param_fn(k, p):
            p = p.clone()
            if k == "opacities":
                p[src] = op_new
            elif k == "scales":
                p[src] = sc_new
            return torch.cat([p, p[src]])
        _rewrite(params, optimizers, state, param_fn,
                 lambda k, v: torch.cat([v, v.new_zeros((n, *v.shape[1:]))]))
        return n

    def _noise(self, params, lr: float):
        from src.gaussians import covariance_3d
        o = torch.sigmoid(params["opacities"])
        gate = torch.sigmoid(100 * ((1 - o) - 0.995))
        Sigma = covariance_3d(params["quats"], torch.exp(params["scales"]))
        eps = torch.randn_like(params["means"]) * gate[:, None] * (lr * self.noise_lr)
        params["means"].add_(torch.einsum("nij,nj->ni", Sigma, eps))
