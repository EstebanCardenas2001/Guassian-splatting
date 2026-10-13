"""Stage 9: PLY export - round trip, agreement with gsplat's exporter, identical renders."""
import numpy as np
import pytest
import torch

from src.export import ply_property_names, read_ply, write_ply
from src.gaussians import Gaussians

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def model(N=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return Gaussians(r(N, 3), r(N, 4), r(N, 3) - 3, r(N), sh0=r(N, 1, 3), shN=0.1 * r(N, 15, 3))


def test_layout_and_round_trip(tmp_path):
    m = model()
    n = write_ply(m, tmp_path / "a.ply")
    assert n == 500
    header = (tmp_path / "a.ply").read_bytes().split(b"end_header\n")[0].decode()
    props = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
    assert props == ply_property_names(15) and len(props) == 62          # standard 3DGS (SH 3)
    r = read_ply(tmp_path / "a.ply")
    for k in m.params:
        assert torch.equal(r.params[k], m.params[k].float()), k


def test_matches_gsplat_exporter(tmp_path):
    """Every property gsplat writes must equal mine (gsplat omits the zero normals)."""
    from gsplat.exporter import export_splats
    from plyfile import PlyData
    m = model(seed=1)
    p = {k: v.detach() for k, v in m.params.items()}
    export_splats(p["means"], p["scales"], p["quats"], p["opacities"], p["sh0"], p["shN"],
                  format="ply", save_to=str(tmp_path / "g.ply"))
    write_ply(m, tmp_path / "a.ply")
    a, g = PlyData.read(str(tmp_path / "a.ply"))["vertex"], PlyData.read(str(tmp_path / "g.ply"))["vertex"]
    for prop in g.properties:
        assert np.array_equal(np.asarray(a[prop.name]), np.asarray(g[prop.name])), prop.name
    assert {"nx", "ny", "nz"} <= {q.name for q in a.properties}


def test_opacity_filter_drops_only_invisible(tmp_path):
    m = model(seed=2)
    with torch.no_grad():
        m.params["opacities"][:100] = -10.0                              # opacity 4.5e-5
    assert write_ply(m, tmp_path / "f.ply", min_opacity=1 / 255) == 400


@pytest.mark.skipif(DEV != "cuda", reason="render needs CUDA")
def test_reloaded_model_renders_identically(tmp_path):
    from src.camera import intrinsics_matrix, look_at
    from src.render_gsplat import render
    m = model(seed=3).to(DEV)
    write_ply(m, tmp_path / "a.ply")
    r = read_ply(tmp_path / "a.ply").to(DEV)
    V = look_at(torch.tensor([0.0, 1.0, -6.0]), torch.zeros(3), torch.tensor([0.0, 1.0, 0.0])).to(DEV)
    K = intrinsics_matrix(100.0, 100.0, 64.0, 48.0).to(DEV)
    a = render(*m.geometry(), m.colors_for(V), V, K, 128, 96)[0]
    b = render(*r.geometry(), r.colors_for(V), V, K, 128, 96)[0]
    assert torch.equal(a, b)
