"""Stage 7: spherical harmonics - orthonormality, gsplat agreement, degree-0 round trip."""
import math

import pytest
import torch

from src.sh import C0, num_coeffs, rgb_to_sh0, sh_basis, sh_to_color

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def random_dirs(n, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=dtype), dim=-1)


def test_basis_is_orthonormal_on_the_sphere():
    """Monte Carlo: (4 pi / M) sum_d Y_i(d) Y_j(d) -> delta_ij for uniform directions."""
    d = random_dirs(2_000_000)
    Y = sh_basis(d, 3)                                           # (M, 16)
    G = 4 * math.pi * (Y.T @ Y) / len(d)
    assert torch.allclose(G, torch.eye(16, dtype=G.dtype), atol=6e-3), (G - torch.eye(16)).abs().max()


def test_degree0_round_trip():
    rgb = torch.rand(100, 3, dtype=torch.float64)
    coeffs = rgb_to_sh0(rgb)[:, None, :]
    assert torch.allclose(sh_to_color(coeffs, random_dirs(100), 0), rgb)
    assert abs(C0 - 0.5 / math.sqrt(math.pi)) < 1e-15


@pytest.mark.skipif(DEV != "cuda", reason="gsplat needs CUDA")
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_matches_gsplat(degree):
    from gsplat import spherical_harmonics
    n = 1000
    d = random_dirs(n, seed=degree, dtype=torch.float32).to(DEV)
    coeffs = torch.randn(n, 16, 3, generator=torch.Generator().manual_seed(1)).to(DEV)
    ref = spherical_harmonics(degree, d, coeffs)                 # gsplat: sum k Y, no offset
    mine = (sh_basis(d, degree)[..., None] * coeffs[:, :num_coeffs(degree)]).sum(-2)
    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max()
