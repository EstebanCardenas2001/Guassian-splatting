"""Stage 5: my adaptive density control vs gsplat's DefaultStrategy, plus its own rules."""
import pytest
import torch

from src.densify import ADC

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
DEV = "cuda"


def make_state(N=400, seed=0):
    """Params with mixed sizes/opacities, Adam moments populated, and a packed 'info' whose
    means2d gradient gives a spread of view-space gradient magnitudes."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device=DEV)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(r(N, 3)),
        "quats": torch.nn.Parameter(r(N, 4)),
        "scales": torch.nn.Parameter(-4.5 + 1.5 * r(N, 3)),          # straddles the clone/split cut
        "opacities": torch.nn.Parameter(2.0 * r(N)),
        "colors": torch.nn.Parameter(r(N, 3))})
    opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    sum(p.square().sum() for p in params.values()).backward()
    for o in opts.values():
        o.step(); o.zero_grad()
    ids = torch.randperm(N, generator=g, device=DEV)[: N * 3 // 4].sort().values   # visible subset
    means2d = (r(len(ids), 2) * 1e-6).requires_grad_()
    (means2d * r(len(ids), 2) * 3e-5 * torch.rand(len(ids), 1, generator=g, device=DEV)).sum().backward()
    info = {"width": 800, "height": 600, "n_cameras": 1, "gaussian_ids": ids,
            "radii": torch.ones(len(ids), 2, device=DEV), "means2d": means2d}
    return params, opts, info


def clone_state(params, opts):
    p2 = torch.nn.ParameterDict({k: torch.nn.Parameter(v.detach().clone()) for k, v in params.items()})
    o2 = {}
    for k, o in opts.items():
        o2[k] = torch.optim.Adam([p2[k]], lr=1e-3)
        o2[k].state[p2[k]] = {kk: vv.clone() for kk, vv in o.state[params[k]].items()}
    return p2, o2


def test_matches_gsplat_default_strategy_exactly():
    from gsplat.strategy import DefaultStrategy
    params, opts, info = make_state()
    p_ref, o_ref = clone_state(params, opts)
    mine, ref = ADC(opacity_reset=False), DefaultStrategy()
    s_mine, s_ref = mine.initialize_state(scene_scale=2.0), ref.initialize_state(scene_scale=2.0)
    for step in (599, 600):                         # accumulate, then refine at a refine_every step
        torch.manual_seed(123)
        mine.step_post_backward(params, opts, s_mine, step, info, packed=True)
        torch.manual_seed(123)
        ref.step_post_backward(p_ref, o_ref, s_ref, step, info, packed=True)
    n_clone, n_split, n_prune = s_mine["log"][-1][1:4]
    assert n_clone > 0 and n_split > 0 and n_prune > 0, s_mine["log"]
    for k in params:
        assert params[k].shape == p_ref[k].shape, k
        assert torch.equal(params[k], p_ref[k]), k
        for m in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(opts[k].state[params[k]][m], o_ref[k].state[p_ref[k]][m]), (k, m)
        assert opts[k].param_groups[0]["params"][0] is params[k]


def test_opacity_reset_caps_and_zeroes_moments():
    params, opts, info = make_state()
    adc = ADC(refine_start_iter=10**9)
    st = adc.initialize_state(scene_scale=2.0)
    adc.step_post_backward(params, opts, st, 3000, info)
    o = torch.sigmoid(params["opacities"])
    assert o.max() <= 0.01 + 1e-6
    assert opts["opacities"].state[params["opacities"]]["exp_avg"].abs().max() == 0


def test_split_children_follow_the_parent_distribution():
    """A large Gaussian split many times: children means ~ N(mu, Sigma), scales / 1.6."""
    N = 20000
    q = torch.nn.functional.normalize(torch.tensor([[0.9, 0.1, 0.3, 0.2]], device=DEV), dim=-1)
    s = torch.tensor([[0.5, 0.2, 0.1]], device=DEV)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros(N, 3, device=DEV)),
        "quats": torch.nn.Parameter(q.repeat(N, 1)),
        "scales": torch.nn.Parameter(torch.log(s).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.full((N,), 2.0, device=DEV))})
    opts = {k: torch.optim.Adam([p]) for k, p in params.items()}
    adc = ADC(prune_scale3d=1e9)
    st = adc.initialize_state(scene_scale=1.0)
    st["grad2d"], st["count"] = torch.ones(N, device=DEV), torch.ones(N, device=DEV)
    torch.manual_seed(0)
    n_clone, n_split = adc._grow(params, opts, st)
    assert (n_clone, n_split) == (0, N) and len(params["means"]) == 2 * N
    from src.gaussians import covariance_3d
    Sigma = covariance_3d(q, s)[0]
    emp = torch.cov(params["means"].T)
    assert torch.allclose(emp, Sigma, atol=3e-3), (emp, Sigma)
    assert torch.allclose(torch.exp(params["scales"]), (s / 1.6).expand(2 * N, 3))
