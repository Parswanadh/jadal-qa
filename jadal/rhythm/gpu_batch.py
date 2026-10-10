"""GPU-resident training loop: keep the whole dataset on the GPU for the fold.

Why this exists
---------------
The first version of `eval/rhythm/train.py` assembled every batch on the host
(`Xtr_t[idx].unsqueeze(0).repeat(...)`) and drew mixup lambdas with
`numpy.random.beta`, so each step paid a host->device copy plus a Python/NumPy
round trip. On this machine that pinned one core at ~100% while the GPU sat
around 78% utilisation: CPU-bound, exactly the wrong way round.

This module keeps the fold's tensors resident in VRAM and draws all randomness
with `torch` on the device. Per step there is no host->device traffic at all.

Budget: the full gold fold is 373 windows x (100 x 6 x 6 x 6) float32 = 322 MB
plus 373 x 100 x 8 = 1.2 MB for the globals. With K=32 members the augmented
batch is at most K x B x T x G x G x C = 32 x 8 x 100 x 6 x 6 x 6 x 4 B = 22 MB.
That fits comfortably inside the 3.5 GiB VRAM budget; the loader refuses to
move a fold to the GPU if it would exceed the configured cap.

WHAT IS NOT DONE:
  * No CUDA graphs, no AMP, no channels-last. Plain float32.
  * The dataset is moved once per fold, so a 7-fold run moves it 7 times.
    That is deliberate: it keeps peak VRAM per process small.
  * If VRAM is unavailable the caller falls back to the host path in train.py.
"""
from __future__ import annotations

import numpy as np
import torch

BYTES_PER_FLOAT = 4


def fold_bytes(X: np.ndarray, g: np.ndarray) -> int:
    return X.size * BYTES_PER_FLOAT + g.size * BYTES_PER_FLOAT


def to_device(X: np.ndarray, g: np.ndarray, device: torch.device,
              cap_gib: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a whole fold to the device once, refusing if it would blow the cap."""
    need = fold_bytes(X, g)
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info()
        cap = int(cap_gib * (1 << 30))
        if need > cap or need > free * 0.6:
            raise MemoryError(
                f"fold needs {need/1e6:.0f} MB but cap is {cap/1e6:.0f} MB "
                f"and {free/1e6:.0f} MB is free; refusing to move it to the GPU")
    Xt = torch.as_tensor(np.ascontiguousarray(X), device=device)
    gt = torch.as_tensor(np.ascontiguousarray(g), device=device)
    return Xt, gt


def rand_beta(shape, alpha: float, device, generator=None) -> torch.Tensor:
    """Beta(alpha, alpha) on the device. Replaces np.random.beta in the hot loop.

    Two Gamma(alpha, 1) draws: X/(X+Y) ~ Beta(alpha, alpha). The gamma draw uses
    the Marsaglia-Tsang method, which is a handful of elementwise ops -- far
    cheaper than a host round trip per step.
    """
    def gamma(a: float, dev, gen):
        # Boost for a < 1
        boost = 1.0
        aa = a
        if a < 1.0:
            boost = torch.rand(shape, device=dev, generator=gen).clamp_min(1e-12) ** (1.0 / a)
            aa = a + 1.0
        d = aa - 1.0 / 3.0
        c = 1.0 / (9.0 * d) ** 0.5
        while True:
            x = torch.randn(shape, device=dev, generator=gen)
            v = (1.0 + c * x) ** 3
            if bool((v > 0).all()):
                break
        u = torch.rand(shape, device=dev, generator=gen)
        out = d * v
        accept = u < 1.0 - 0.0331 * x ** 4
        accept |= torch.log(u.clamp_min(1e-12)) < 0.5 * x ** 2 + d * (1.0 - v + torch.log(v.clamp_min(1e-12)))
        out = torch.where(accept, out, d * v)
        return out * boost

    x = gamma(alpha, device, generator)
    y = gamma(alpha, device, generator)
    return (x / (x + y).clamp_min(1e-12)).clamp(1e-4, 1 - 1e-4)


def mixup_device(X, g, y, alpha: float, generator=None):
    """Mixup with all randomness drawn on the device."""
    K, B = X.shape[0], X.shape[1]
    lam = rand_beta((K, B), alpha, X.device, generator)
    perm = torch.randperm(B, device=X.device, generator=generator)
    lam_x = lam[:, :, None, None, None, None]
    Xm = X * lam_x + X[:, perm] * (1 - lam_x)
    lam_g = lam[:, :, None, None]
    gm = g * lam_g + g[:, perm] * (1 - lam_g)
    ym = y * lam + y[:, perm] * (1 - lam)
    return Xm, gm, ym


def batch_view(X: torch.Tensor, g: torch.Tensor, idx: torch.Tensor, K: int):
    """Gather a batch and replicate it across members, entirely on the device."""
    Xb = X[idx].unsqueeze(0).expand(K, *([-1] * X[idx].dim())).contiguous()
    gb = g[idx].unsqueeze(0).expand(K, *([-1] * g[idx].dim())).contiguous()
    return Xb, gb
