"""Stage 0 sanity check: CUDA from PyTorch, nvcc, gsplat import, COLMAP binary."""
import shutil
import subprocess
import time

import torch

print(f"torch {torch.__version__}  (built for CUDA {torch.version.cuda})")
assert torch.cuda.is_available(), "CUDA not visible from PyTorch"
dev = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(dev)}, capability {torch.cuda.get_device_capability(dev)}")
print(f"arch list in torch build: {torch.cuda.get_arch_list()}")

# FP32 matmul throughput as a smoke test (T4 peak is ~8 TFLOPS FP32).
n = 4096
a, b = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
torch.backends.cuda.matmul.allow_tf32 = False
for _ in range(3):
    a @ b
torch.cuda.synchronize()
t = time.perf_counter()
reps = 10
for _ in range(reps):
    a @ b
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / reps
print(f"fp32 matmul {n}^3: {2 * n**3 / dt / 1e12:.2f} TFLOPS")

nvcc = shutil.which("nvcc")
print(f"nvcc: {nvcc}")
if nvcc:
    print(subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1])

try:
    import gsplat  # noqa: F401
    from gsplat import rasterization  # noqa: F401
    print(f"gsplat {gsplat.__version__} imported")
except Exception as e:  # report, don't hide
    print(f"gsplat import FAILED: {e!r}")

colmap = shutil.which("colmap")
print(f"colmap: {colmap or 'NOT FOUND'}")
