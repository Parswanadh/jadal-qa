"""GPU JPEG/video-frame decode + thumbnail (nvjpeg rule, orchestrator 16:43).

nvjpeg (torchvision decode_jpeg on cuda) + cuda bilinear resize so the
longest side is MAX_SIDE, then to CPU PIL for the EG2 processor.
Pixel output DIFFERS slightly from the PIL path (CPU decode + PIL BILINEAR
thumbnail); the gap is measured in the report (cosine distance on same
frames). Aspect ratio is preserved in both paths.
"""
from __future__ import annotations

import torch
import torchvision
from torchvision.io import ImageReadMode, decode_jpeg
import torch.nn.functional as F

MAX_SIDE = 512


def _cuda_thumb(chw_cuda: torch.Tensor) -> "object":
    """(3,H,W) cuda uint8 -> CPU PIL thumbnailed to longest side MAX_SIDE."""
    from PIL import Image

    _, h, w = chw_cuda.shape
    s = MAX_SIDE / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = F.interpolate(chw_cuda[None].float(), size=(nh, nw), mode="bilinear",
                          align_corners=False)[0].clamp(0, 255).byte()
    return Image.fromarray(small.permute(1, 2, 0).cpu().numpy(), mode="RGB")


def gpu_thumb_jpeg(raw: bytes):
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    img = decode_jpeg(t, device="cuda", mode=ImageReadMode.RGB)  # (3,H,W) cuda
    return _cuda_thumb(img)


def gpu_thumb_bgr(bgr):
    """cv2 BGR uint8 ndarray -> CPU PIL thumb (via GPU)."""
    import numpy as np

    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    t = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1).contiguous()
    return _cuda_thumb(t)
