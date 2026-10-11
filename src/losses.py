r"""Photometric losses.

SSIM (Wang et al. 2004) with an 11x11 Gaussian window (sigma = 1.5), per channel:
    SSIM = ((2 mu_x mu_y + C1)(2 sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)),
    C1 = (0.01 L)^2, C2 = (0.03 L)^2 with dynamic range L = 1,
where mu, sigma^2 and sigma_xy are Gaussian-weighted local moments. Like 3DGS (and
unlike torchmetrics' default), borders use zero padding so the map has the image size.

Training loss as in 3DGS:   L = (1 - lambda) L1 + lambda (1 - SSIM),   lambda = 0.2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(img: Tensor, ref: Tensor, window: int = 11, sigma: float = 1.5, padding: str = "same") -> Tensor:
    """Mean SSIM of (H, W, C) or (B, H, W, C) images in [0, 1].

    padding="same": zero-padded, full-size map (3DGS); "valid": only fully covered pixels
    (matches torchmetrics / skimage with gaussian weights).
    """
    if img.dim() == 3:
        img, ref = img[None], ref[None]
    x, y = img.permute(0, 3, 1, 2), ref.permute(0, 3, 1, 2)          # (B, C, H, W)
    C = x.shape[1]
    w = _gaussian_window(window, sigma, C, x.device, x.dtype)
    pad = window // 2 if padding == "same" else 0
    conv = lambda t: F.conv2d(t, w, padding=pad, groups=C)
    mu_x, mu_y = conv(x), conv(y)
    sxx = conv(x * x) - mu_x * mu_x
    syy = conv(y * y) - mu_y * mu_y
    sxy = conv(x * y) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sxx + syy + C2))
    return s.mean()


def l1(img: Tensor, ref: Tensor) -> Tensor:
    return (img - ref).abs().mean()


def photometric_loss(img: Tensor, ref: Tensor, lambda_dssim: float = 0.2) -> Tensor:
    """(1 - lambda) L1 + lambda (1 - SSIM)."""
    return (1 - lambda_dssim) * l1(img, ref) + lambda_dssim * (1 - ssim(img, ref))


def psnr(img: Tensor, ref: Tensor) -> Tensor:
    """PSNR in dB for images in [0, 1] (img is clamped first)."""
    return -10 * torch.log10(torch.mean((img.clamp(0, 1) - ref) ** 2))
