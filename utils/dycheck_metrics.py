"""
DyCheck-protocol masked image metrics: mPSNR, mSSIM, mLPIPS.

This is a faithful port of the masked-metric math from Shape-of-Motion
(vye16/shape-of-motion, `flow3d/metrics.py`), which is itself a PyTorch
reimplementation of the official DyCheck toolkit
(KAIR-BAIR/dycheck, `dycheck/core/metrics/image.py`).

Why vendored verbatim: MoSca and Dynamic Gaussian Marbles report their DyCheck
numbers under exactly this evaluation recipe. Computing OUR masked metrics with
the same code is what makes our table directly comparable to their published
results, so the math here is deliberately copied rather than "improved".

Tensor conventions (kept identical to Shape-of-Motion to avoid porting drift):
    preds, targets : (B, H, W, 3)  float32, values in [0, 1]
    masks          : (B, H, W)     float32, values in {0, 1}  (covisibility)

mPSNR : masked-mean of MSE over covisible pixels (x3 for RGB), then -10*log10.
mSSIM : SSIM with **partial-convolution** Gaussian windows (11, sigma=1.5) under
        'valid' padding, so out-of-mask pixels do not bleed across the window.
mLPIPS: AlexNet LPIPS with spatial=True; images are mask-composited, the per-pixel
        LPIPS map is then averaged over the covisible mask. normalize=True maps
        [0,1] -> [-1,1] (equivalent to DyCheck's im2tensor(factor=1/2)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_mpsnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> float:
    """Masked PSNR. preds/targets: (B,H,W,3) in [0,1]; masks: (B,H,W) in {0,1}."""
    if masks is None:
        masks = torch.ones_like(preds[..., 0])
    return (
        -10.0
        * torch.log(
            F.mse_loss(
                preds * masks[..., None],
                targets * masks[..., None],
                reduction="sum",
            )
            / masks.sum().clamp(min=1.0)
            / 3.0
        )
        / np.log(10.0)
    ).item()


def compute_mssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Masked SSIM with partial-convolution Gaussian windows ('valid' padding)."""
    if masks is None:
        masks = torch.ones_like(preds[..., 0])

    # 1D Gaussian filter (matches DyCheck/SoM offset convention).
    hw = kernel_size // 2
    shift = (2 * hw - kernel_size + 1) / 2
    f_i = ((torch.arange(kernel_size, device=preds.device) - hw + shift) / sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt = filt / torch.sum(filt)

    def convolve2d(z, m, f):
        # z: (B,H,W,C), m: (B,H,W), f: (Hf,Wf). Partial conv: only valid pixels
        # contribute and the result is renormalised by the valid count.
        z = z.permute(0, 3, 1, 2)
        m = m[:, None]
        f = f[None, None].expand(z.shape[1], -1, -1, -1)
        z_ = F.conv2d(z * m, f, padding="valid", groups=z.shape[1])
        m_ = F.conv2d(m, torch.ones_like(f[:1]), padding="valid")
        out = torch.where(
            m_ != 0,
            z_ * torch.ones_like(f).sum() / (m_ * z.shape[1]),
            torch.zeros_like(z_),
        ).permute(0, 2, 3, 1)
        return out, (m_ != 0)[:, 0].to(z.dtype)

    filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
    filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])
    # Separable blur: 1xK then Kx1 (faster than the full 2D conv).
    filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

    mu0 = filt_fn(preds, masks)[0]
    mu1 = filt_fn(targets, masks)[0]
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(preds ** 2, masks)[0] - mu00
    sigma11 = filt_fn(targets ** 2, masks)[0] - mu11
    sigma01 = filt_fn(preds * targets, masks)[0] - mu01

    # Clip variances/covariances to valid values.
    sigma00 = sigma00.clamp(min=0.0)
    sigma11 = sigma11.clamp(min=0.0)
    sigma01 = torch.sign(sigma01) * torch.minimum(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    return ssim_map.mean(dim=(1, 2, 3)).mean().item()


@torch.no_grad()
def compute_mlpips(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lpips_model,
    masks: torch.Tensor | None = None,
) -> float:
    """Masked LPIPS. `lpips_model` must be lpips.LPIPS(net='alex', spatial=True)."""
    if masks is None:
        masks = torch.ones_like(preds[..., 0])
    scores = lpips_model(
        (preds * masks[..., None]).permute(0, 3, 1, 2),
        (targets * masks[..., None]).permute(0, 3, 1, 2),
        normalize=True,
    )  # (B, 1, H, W) spatial LPIPS map
    sum_scores = (scores * masks[:, None]).sum()
    total = masks.sum().clamp(min=1.0)
    return (sum_scores / total).item()
