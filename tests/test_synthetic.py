"""Stage 3: ground-truth scene generation and parameter-recovery metrics."""
import math

import torch

from src.gaussians import covariance_3d
from src.synthetic import compare_parameters, make_gt_scene


def _perm_axes(quats, scales):
    """Same covariance, different (q, s): rotate the frame by +90 deg about local z, which
    swaps the x/y axes, and swap the scales accordingly; also flip the quaternion sign."""
    h = math.sqrt(0.5)
    rz = torch.tensor([h, 0.0, 0.0, h], dtype=quats.dtype)
    w1, x1, y1, z1 = quats.unbind(-1)
    w2, x2, y2, z2 = rz
    q = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,       # Hamilton product q * rz
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
    return -q, scales[:, [1, 0, 2]]


def test_axis_permutation_gives_same_covariance():
    gt = make_gt_scene(50, seed=0)
    q2, s2 = _perm_axes(gt["quats"], gt["scales"])
    assert torch.allclose(covariance_3d(gt["quats"], gt["scales"]), covariance_3d(q2, s2), atol=1e-6)
    assert (torch.nn.functional.normalize(q2, dim=-1) - gt["quats"]).abs().max() > 0.1


def test_perfect_recovery_is_invariant_to_order_sign_and_axis_permutation():
    gt = make_gt_scene(50, seed=0)
    perm = torch.randperm(50, generator=torch.Generator().manual_seed(1))
    q2, s2 = _perm_axes(gt["quats"][perm], gt["scales"][perm])
    rec = {"means": gt["means"][perm], "quats": q2, "scales": s2,
           "opacities": gt["opacities"][perm], "colors": gt["colors"][perm]}
    m = compare_parameters(gt, rec)
    assert m["pos_err"].max() < 1e-6
    assert m["cov_rel_err"].max() < 1e-5
    assert m["axis_err_deg"].max() < 0.1
    assert m["opacity_err"].max() < 1e-6 and m["color_err"].max() < 1e-6
    assert m["recovered_frac"] == 1.0
    # ...whereas a naive quaternion comparison would call these rotations wrong
    assert m["naive_quat_err_deg"].min() > 10


def test_recovery_metrics_detect_errors():
    gt = make_gt_scene(50, seed=0)
    rec = {k: v.clone() for k, v in gt.items()}
    rec["means"][:10] += 0.2
    m = compare_parameters(gt, rec)
    assert m["recovered_frac"] <= 0.8 + 1e-9
