"""Tiny gsplat render: triggers the JIT build of gsplat's CUDA kernels and checks forward + backward."""
import torch
from gsplat import rasterization

d, N = "cuda", 100
torch.manual_seed(0)
means = (torch.randn(N, 3, device=d) * 0.3).requires_grad_()
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=d), dim=-1)
scales = torch.full((N, 3), 0.05, device=d)
opac = torch.full((N,), 0.8, device=d)
cols = torch.rand(N, 3, device=d)
viewmat = torch.eye(4, device=d)[None]
viewmat[0, 2, 3] = 3.0  # camera 3 units behind the origin, looking down +z
K = torch.tensor([[[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]]], device=d)
img, alpha, _ = rasterization(means, quats, scales, opac, cols, viewmat, K, 128, 128)
img.sum().backward()
print(f"render {tuple(img.shape)}, alpha max {alpha.max().item():.3f}, "
      f"|grad means| {means.grad.norm().item():.3e}")
