"""Sustained T4 throughput: FP32 / FP16-tensor-core matmul and device memory bandwidth.

Each test runs for ``--seconds`` so the GPU reaches its power/thermal steady state
(short bursts overstate what a 30k-iteration training run will see). Run alongside
`nvidia-smi dmon -s pc` to see clocks and power while it runs.
"""
import argparse
import time

import torch


def timed(fn, seconds):
    """Call ``fn`` repeatedly for ~``seconds``; return mean seconds per call (after warm-up)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
        if n % 10 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15)
    args = ap.parse_args()
    d = torch.device("cuda")

    n = 8192
    for dtype, peak, label in [(torch.float32, 8.1, "fp32 (CUDA cores)"),
                               (torch.float16, 65.0, "fp16 (tensor cores)")]:
        a, b = torch.randn(n, n, device=d, dtype=dtype), torch.randn(n, n, device=d, dtype=dtype)
        dt = timed(lambda: a @ b, args.seconds)
        tf = 2 * n**3 / dt / 1e12
        print(f"{label:22s} {tf:6.2f} TFLOPS  ({100 * tf / peak:4.1f}% of {peak} spec)")
        del a, b

    x = torch.empty(256 * 2**20, device=d, dtype=torch.float32)  # 1 GiB
    y = torch.empty_like(x)
    dt = timed(lambda: y.copy_(x), args.seconds)
    gbs = 2 * x.numel() * 4 / dt / 1e9  # read + write
    print(f"{'device copy':22s} {gbs:6.1f} GB/s     ({100 * gbs / 320:4.1f}% of 320 spec)")


if __name__ == "__main__":
    main()
