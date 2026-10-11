"""Stage 2: 3D forward model - rotations, covariance, cameras, projection, EWA, culling,
depth-sorted compositing, multi-view consistency and gradcheck."""
import math

import pytest
import torch

from src.camera import intrinsics_matrix, look_at
from src.gaussians import covariance_3d, quat_to_rotmat
from src.rasterize import rasterize_dense
from src.render_torch import (ewa_covariance_2d, project_gaussians, project_points, render,
                              world_to_camera)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
F64 = dict(dtype=torch.float64, device=DEV)


def rand_quats(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4, generator=g, dtype=torch.float64).to(DEV)


def simple_camera(W=64, H=48, f=60.0, eye=(0.0, 0.0, -5.0)):
    K = intrinsics_matrix(f, f, W / 2, H / 2).to(**F64)
    V = look_at(torch.tensor(eye, **F64), torch.zeros(3, **F64), torch.tensor([0.0, 1.0, 0.0], **F64))
    return V, K, W, H


# ---------------------------------------------------------------- rotations & covariance
def test_quat_identity_and_rotation_about_z():
    I = quat_to_rotmat(torch.tensor([1.0, 0, 0, 0], **F64))
    assert torch.allclose(I, torch.eye(3, **F64))
    h = math.sqrt(0.5)
    Rz = quat_to_rotmat(torch.tensor([h, 0, 0, h], **F64))       # +90 deg about z
    assert torch.allclose(Rz @ torch.tensor([1.0, 0, 0], **F64), torch.tensor([0.0, 1, 0], **F64))


def test_quat_is_scale_and_sign_invariant_and_orthonormal():
    q = rand_quats(32)
    R = quat_to_rotmat(q)
    assert torch.allclose(R, quat_to_rotmat(3.7 * q))
    assert torch.allclose(R, quat_to_rotmat(-q))
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, **F64).expand(32, 3, 3), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(32, **F64))


def test_covariance_eigenvalues_are_squared_scales():
    q = rand_quats(16, seed=1)
    s = torch.rand(16, 3, **F64) + 0.1
    S = covariance_3d(q, s)
    assert torch.allclose(S, S.transpose(-1, -2))
    ev = torch.linalg.eigvalsh(S)
    assert torch.allclose(ev, torch.sort(s ** 2, dim=-1).values, atol=1e-12)


# ---------------------------------------------------------------- cameras & projection
def test_look_at_opencv_convention():
    """Camera on -z looking at the origin, world +y up: origin is straight ahead (z>0),
    world +y appears at negative camera y (top of the image). Looking along +z with +y up
    in a right-handed world, world +x is on the camera's *left* (x_cam = -up x forward)."""
    V, K, W, H = simple_camera()
    p = world_to_camera(torch.tensor([[0.0, 0, 0], [0, 1, 0], [1, 0, 0]], **F64), V)
    assert torch.allclose(p[0], torch.tensor([0.0, 0, 5], **F64))
    assert p[1, 1] < 0 and p[2, 0] < 0
    R = V[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3, **F64)) and torch.isclose(torch.linalg.det(R), torch.tensor(1.0, **F64))


def test_pinhole_projection():
    K = intrinsics_matrix(100.0, 120.0, 32.0, 24.0).to(**F64)
    p = torch.tensor([[0.0, 0, 2], [1.0, -0.5, 4]], **F64)
    uv = project_points(p, K)
    assert torch.allclose(uv[0], torch.tensor([32.0, 24], **F64))
    assert torch.allclose(uv[1], torch.tensor([100 * 0.25 + 32, 120 * -0.125 + 24], **F64))


def test_ewa_equals_autograd_jacobian():
    """Sigma2D = J W Sigma W^T J^T + eps I, with J W the Jacobian of world -> pixel."""
    V, K, W, H = simple_camera()
    means = torch.tensor([[0.3, -0.2, 0.5], [-0.4, 0.1, -1.0]], **F64)
    cov = covariance_3d(rand_quats(2, seed=2), torch.tensor([[0.1, 0.2, 0.05], [0.3, 0.1, 0.2]], **F64))
    got = ewa_covariance_2d(cov, world_to_camera(means, V), V, K, W, H, eps2d=0.3)
    for i in range(2):
        Jw = torch.autograd.functional.jacobian(
            lambda x: project_points(world_to_camera(x[None], V), K)[0], means[i])   # (2, 3)
        want = Jw @ cov[i] @ Jw.T + 0.3 * torch.eye(2, **F64)
        assert torch.allclose(got[i], want, atol=1e-10)


def test_ewa_matches_monte_carlo_for_small_gaussian():
    """For a Gaussian small relative to its depth, projecting samples and taking their
    covariance reproduces the linearized (EWA) covariance."""
    V, K, W, H = simple_camera()
    mean = torch.tensor([0.2, 0.1, 0.0], **F64)
    cov = covariance_3d(rand_quats(1, seed=3), torch.tensor([[0.02, 0.05, 0.01]], **F64))[0]
    g = torch.Generator(device=DEV).manual_seed(0)
    x = mean + torch.randn(400_000, 3, generator=g, **F64) @ torch.linalg.cholesky(cov).T
    uv = project_points(world_to_camera(x, V), K)
    emp = torch.cov(uv.T)
    ewa = ewa_covariance_2d(cov[None], world_to_camera(mean[None], V), V, K, W, H, eps2d=0.0)[0]
    assert torch.allclose(emp, ewa, rtol=0.02, atol=1e-3)


# ---------------------------------------------------------------- culling
def test_gaussians_behind_or_too_close_are_culled():
    V, K, W, H = simple_camera()
    # camera at z=-5 looking +z: world z=-6 is behind, z=-4.995 is closer than near=0.01
    means = torch.tensor([[0.0, 0, -6.0], [0, 0, -4.995], [0, 0, 0], [50, 0, 0]], **F64)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 4, **F64)
    scales = torch.full((4, 3), 0.05, **F64)
    proj = project_gaussians(means, quats, scales, V, K, W, H, near=0.01)
    assert proj["valid"].tolist() == [False, False, True, False]   # last: far outside frustum
    assert (proj["radii"][~proj["valid"]] == 0).all()

    bg = torch.tensor([0.2, 0.4, 0.6], **F64)
    img, alpha, info = render(means[:2], quats[:2], scales[:2], torch.ones(2, **F64),
                              torch.ones(2, 3, **F64), V, K, W, H, bg=bg)
    assert info["pairs"] == 0 and torch.allclose(img, bg.expand(H, W, 3)) and alpha.abs().max() == 0


# ---------------------------------------------------------------- compositing
def test_single_gaussian_center_pixel():
    """Mean projected exactly onto a pixel center: C = o c + (1 - o) bg there."""
    V, K, W, H = simple_camera(W=65, H=49, f=60.0)        # principal point at a pixel center
    K[0, 2], K[1, 2] = 32.5, 24.5
    o, c, bg = 0.7, torch.tensor([1.0, 0.5, 0.0], **F64), torch.tensor([0.0, 0.0, 1.0], **F64)
    img, alpha, _ = render(torch.zeros(1, 3, **F64), torch.tensor([[1.0, 0, 0, 0]], **F64),
                           torch.full((1, 3), 0.2, **F64), torch.tensor([o], **F64), c[None],
                           V, K, W, H, bg=bg)
    assert torch.allclose(img[24, 32], o * c + (1 - o) * bg, atol=1e-12)
    assert torch.isclose(alpha[24, 32], torch.tensor(o, **F64))


@pytest.mark.parametrize("swap_index_order", [False, True])
def test_front_gaussian_wins_regardless_of_index_order(swap_index_order):
    V, K, W, H = simple_camera()
    means = torch.tensor([[0.0, 0, -1.0], [0, 0, 1.0]], **F64)   # red in front (closer to z=-5)
    colors = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]], **F64)
    if swap_index_order:
        means, colors = means.flip(0), colors.flip(0)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, **F64)
    scales = torch.full((2, 3), 0.5, **F64)
    img, _, _ = render(means, quats, scales, torch.full((2,), 0.95, **F64), colors, V, K, W, H)
    center = img[H // 2, W // 2]
    assert center[0] > 0.9 and center[2] < 0.06


def test_sparse_render_matches_dense_reference():
    V, K, W, H = simple_camera(W=40, H=30, f=40.0)
    g = torch.Generator().manual_seed(4)
    N = 40
    means = ((torch.rand(N, 3, generator=g, dtype=torch.float64) - 0.5) * 3).to(DEV)
    quats = rand_quats(N, seed=5)
    scales = (0.05 + 0.3 * torch.rand(N, 3, generator=g, dtype=torch.float64)).to(DEV)
    opac = (0.2 + 0.79 * torch.rand(N, generator=g, dtype=torch.float64)).to(DEV)
    colors = torch.rand(N, 3, generator=g, dtype=torch.float64).to(DEV)
    bg = torch.tensor([0.1, 0.2, 0.3], **F64)
    for alpha_min in (0.0, 1 / 255):
        img, alpha, _ = render(means, quats, scales, opac, colors, V, K, W, H, bg=bg,
                               alpha_min=alpha_min)
        p = project_gaussians(means, quats, scales, V, K, W, H)
        ref_img, ref_alpha = rasterize_dense(p["means2d"], p["conics"], p["radii"], colors,
                                             opac, H, W, depths=p["depths"], bg=bg,
                                             alpha_min=alpha_min)
        assert torch.allclose(img, ref_img, atol=1e-10)
        assert torch.allclose(alpha, ref_alpha, atol=1e-10)


def test_hand_placed_gaussian_lands_where_expected_from_several_views():
    """A small bright Gaussian at P: in every view, the brightest pixel is the one that
    contains K [R | t] P (computed here by hand, without the renderer's helpers)."""
    P = torch.tensor([0.4, -0.3, 0.2], **F64)
    K = intrinsics_matrix(80.0, 80.0, 40.0, 30.0).to(**F64)
    W, H = 80, 60
    for eye in ([0, 0, -4.0], [4.0, 0, 0], [0, -4.0, 0.1], [-2.0, 2.0, 3.0]):
        V = look_at(torch.tensor(eye, **F64), torch.zeros(3, **F64), torch.tensor([0.0, 1, 0], **F64))
        pc = V[:3, :3] @ P + V[:3, 3]
        u, v = (K @ (pc / pc[2]))[:2]
        img, _, _ = render(P[None], torch.tensor([[1.0, 0, 0, 0]], **F64),
                           torch.full((1, 3), 0.03, **F64), torch.tensor([0.9], **F64),
                           torch.ones(1, 3, **F64), V, K, W, H)
        i, j = divmod(int(img[..., 0].argmax()), W)
        assert (i, j) == (int(v), int(u)), (eye, (i, j), (float(v), float(u)))


def test_antialiased_mode_compensates_opacity():
    """Mip-Splatting-style: opacity scaled by sqrt(det(S) / det(S + eps I)) <= 1;
    ~1 for large footprints, < 1 for sub-pixel ones."""
    V, K, W, H = simple_camera()
    means = torch.zeros(2, 3, **F64)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, **F64)
    scales = torch.tensor([[1.0] * 3, [0.002] * 3], **F64)
    p = project_gaussians(means, quats, scales, V, K, W, H, antialiased=True)
    comp = p["opacity_scale"]
    assert comp[0] > 0.99 and comp[1] < 0.2


# ---------------------------------------------------------------- gradients
def test_gradcheck_3d():
    V, K, W, H = simple_camera(W=14, H=12, f=14.0)
    g = torch.Generator().manual_seed(6)
    N = 4
    means = ((torch.rand(N, 3, generator=g, dtype=torch.float64) - 0.5) * 1.5).to(DEV)
    quats = rand_quats(N, seed=7)
    scales = (0.2 + 0.3 * torch.rand(N, 3, generator=g, dtype=torch.float64)).to(DEV)
    opac = (0.2 + 0.5 * torch.rand(N, generator=g, dtype=torch.float64)).to(DEV)
    colors = torch.rand(N, 3, generator=g, dtype=torch.float64).to(DEV)
    bg = torch.tensor([0.3, 0.6, 0.9], **F64)
    params = [t.clone().requires_grad_() for t in (means, quats, scales, opac, colors)]
    f = lambda m, q, s, o, c: render(m, q, s, o, c, V, K, W, H, bg=bg, t_min=0.0,
                                     alpha_min=0.0)[0]
    assert render(*params, V, K, W, H)[2]["pairs"] > 50
    assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-6, nondet_tol=1e-12)
