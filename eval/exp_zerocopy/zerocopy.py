"""Z1 zero-copy GPU-path helpers.

Honest scope
------------
The repo's decode stage is an **out-of-process ffmpeg** process whose only output
to the parent is a host-memory pipe (``-f rawvideo ... pipe:1``).  There is no
ffmpeg muxer that writes CUDA device memory, so frames cannot stay in VRAM
across that process boundary.  A genuinely VRAM-resident path needs an
**in-process** NVDEC binding.  This module tries ``torchcodec`` (the only
in-process GPU decoder that can be installed here); when it is unavailable the
caller must fall back to uploading hwdownloaded frames ("partial GPU path") and
that is reported, never hidden.

Everything here is deterministic; no sampling at inference.
"""
from __future__ import annotations

import numpy as np

# EG2 frozen recipe constants (mirror configs/ensemble_v1.json + the model card)
EGW, EGH = 512, 288          # the repo EG2 branch size (scale_cuda target)
PATCH = 16
MAX_SOFT_TOKENS = 140        # configs/ensemble_v1.json views.eg2.max_soft_tokens
POOL_K = 3
EG2_STEP_S = 0.5


def have_torchcodec():
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import VideoDecoder  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def decode_torchcodec(path, start_s, dur_s, device="cuda"):
    """In-process NVDEC decode of [start_s, start_s+dur_s) -> CUDA uint8 frames.

    Returns ``(frames, fps, meta)`` with ``frames`` an ``(T, H, W, 3)`` uint8
    tensor on ``device``.  Raises if torchcodec / CUDA decode is unavailable.
    """
    import torch
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(path, device=device, dimension_order="NCHW")
    md = dec.metadata
    fps = float(md.average_fps)
    fb = dec.get_frames_played_in_range(start_seconds=float(start_s),
                                        stop_seconds=float(start_s) + float(dur_s))
    data = fb.data  # (T, C, H, W) uint8 on device
    if data.numel() == 0:
        raise RuntimeError(f"torchcodec: no frames {path}@{start_s}")
    frames = data.permute(0, 2, 3, 1).contiguous()  # (T,H,W,C)
    meta = {"width": int(md.width), "height": int(md.height), "fps": fps,
            "codec": str(md.codec), "n_frames": int(frames.shape[0])}
    return frames, fps, meta


def luma_of(rgb):
    """rgb: (...,3) float tensor -> (...,) luma."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def gpu_pixel_features(frames_nhwc_u8, fps, w=160, h=90):
    """Pixel-26 via the repo GPU kernels on CUDA frames.

    ``frames_nhwc_u8``: (T,H,W,3) uint8 CUDA.  Resizes on GPU to (h,w), then
    ``jadal.gpu.features.gpu_window_features_batch``.  Returns (feat_dict,
    luma_tensor, rgb_tensor).
    """
    import torch
    import torch.nn.functional as F
    from jadal.gpu.features import gpu_window_features_batch, FEATURE_NAMES

    assert frames_nhwc_u8.is_cuda, "frames must be CUDA-resident"
    T = frames_nhwc_u8.shape[0]
    xs = []
    for i in range(0, T, 64):  # chunked so a native 1080p window never blows VRAM
        chunk = frames_nhwc_u8[i:i + 64].permute(0, 3, 1, 2).float() / 255.0
        xs.append(F.interpolate(chunk, size=(h, w), mode="bilinear",
                                align_corners=False, antialias=True))
    x = torch.cat(xs, dim=0)                                     # (T,3,h,w)
    rgb = x.permute(0, 2, 3, 1).contiguous()                     # (T,h,w,3)
    luma = luma_of(rgb)
    fb = gpu_window_features_batch(luma.unsqueeze(0), rgb.unsqueeze(0),
                                   fps=float(fps))               # (1,26)
    vals = fb[0].detach().to("cpu").numpy().astype(np.float64)
    feats = {k: float(v) for k, v in zip(FEATURE_NAMES, vals)}
    return feats, luma, rgb


def _eg2_text_cache(model, n, device, max_soft_tokens=MAX_SOFT_TOKENS):
    """Build (once) the frozen chat-template input_ids/attention_mask for n frames.

    The EG2 branch always decodes n frames of exactly EGW x EGH, so
    ``num_soft_tokens_per_image`` (hence the image-token count) is identical
    for every window; the text side can be cached per n.
    """
    from PIL import Image
    dummy = [Image.new("RGB", (EGW, EGH), (127, 127, 127)) for _ in range(n)]
    pre = model[0].preprocess(dummy,
                              processing_kwargs={"image": {"max_soft_tokens": max_soft_tokens}})
    ids = pre["input_ids"].to(device)
    am = pre.get("attention_mask")
    am = am.to(device) if am is not None else None
    return ids, am


def gpu_eg2_embed(frames_nhwc_u8, model, fps=30.0, step_s=EG2_STEP_S,
                  cache=None, w=EGW, h=EGH, max_soft_tokens=MAX_SOFT_TOKENS):
    """EG2 L2-normalised mean embedding directly from CUDA frames.

    ``frames_nhwc_u8``: (T,H,W,3) uint8 CUDA (native fps).  Selects ~2 fps
    frames, resizes on GPU to (h,w), builds pixel_values with the repo model's
    own Gemma4 image processor *on the GPU* (no hwdownload), runs the frozen
    EmbeddingGemma-2 forward + mean pooling + L2 normalise.  ``cache`` is a dict
    reused across windows keyed by frame count.
    """
    import torch
    import torch.nn.functional as F
    from transformers.models.gemma4 import image_processing_gemma4 as G4

    dev = frames_nhwc_u8.device
    T = frames_nhwc_u8.shape[0]
    stride = max(int(round(step_s * float(fps))), 1)
    sel = frames_nhwc_u8[::stride]
    n = int(sel.shape[0])
    if n == 0:
        raise RuntimeError("no EG2 frames selected")
    # GPU resize to the repo EG2 branch size (matches scale_cuda=512:288)
    x = sel.permute(0, 3, 1, 2).float()                          # (n,3,H,W)
    x = F.interpolate(x, size=(h, w), mode="bilinear",
                      align_corners=False, antialias=True)       # (n,3,288,512)
    ip = model[0].processor.image_processor
    pv = ip._preprocess(
        images=list(x),
        do_resize=True,
        resample=G4.tvF.InterpolationMode.BICUBIC,
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=False,
        image_mean=None,
        image_std=None,
        return_tensors="pt",
        patch_size=PATCH,
        max_soft_tokens=max_soft_tokens,
        pooling_kernel_size=POOL_K,
    )
    if cache is not None and n in cache:
        ids, am = cache[n]
    else:
        ids, am = _eg2_text_cache(model, n, dev, max_soft_tokens=max_soft_tokens)
        if cache is not None:
            cache[n] = (ids, am)
    feats = {
        "input_ids": ids,
        "attention_mask": am,
        "pixel_values": pv["pixel_values"].to(dev),
        "image_position_ids": pv["image_position_ids"].to(dev),
        "modality": "image",
    }
    feats = model[0](feats)          # -> token_embeddings
    feats = model[1](feats)          # mean pooling
    feats = model[2](feats)          # L2 normalise
    E = feats["sentence_embedding"]
    m = E.mean(dim=0)
    m = m / m.norm().clamp_min(1e-12)
    return m.detach().to("cpu").float().numpy().astype(np.float64), n
