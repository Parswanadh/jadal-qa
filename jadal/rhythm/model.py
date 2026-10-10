"""Vectorised ensemble: K independent members trained in one set of tensor ops.

The ensemble dimension K is carried as a leading tensor dimension and every
weight is stored as (K, ...). Convolutions use `groups=K*N` after folding K into
the batch/channel dimension, so each member keeps its own weights while the GPU
sees one large batched convolution. This avoids `torch.func.vmap` /
`functional_call` and keeps the peak footprint flat and predictable.

Inputs
  X : (K, B, T, G, G, C)   grid channels (see jadal.rhythm.features)
  g : (K, B, T, 8)         global per-frame channels

Outputs
  logit  : (K, B)          usable vs not-usable
  aux    : (K, B, 2)       auxiliary reason logits (NO_ACTIVE_WORK, HANDS_NOT_VISIBLE)

Permutation equivariance: the encoder is applied to every grid cell with shared
weights, then pooled over cells with mean+max, so the model is invariant to any
permutation of the cells (and therefore to camera framing / grid relabelling).

WHAT IS NOT DONE:
  * No spatial (across-cell) convolution. Cells are independent until pooling, so
    the model cannot represent "this cell is next to that cell". A spatial mixing
    layer is the obvious next addition and is NOT implemented here.
  * No attention pooling; mean+max only.
  * No architecture search; sizes come from the config.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-member linear layer.

    x : (K, N, Cin)   w : (K, Cin, Cout)   b : (K, Cout)  ->  (K, N, Cout)

    Written as an einsum over the ensemble dimension: each member k has its own
    weight matrix and applies it to every one of its N samples. This is the
    reference formulation -- it was checked numerically against a per-member
    python loop (tests/test_rhythm_model.py). A grouped-conv rewrite is possible
    but was abandoned because the (groups, in_per_group, out_per_group) channel
    bookkeeping is error-prone and this op is not the bottleneck (see the timing
    in docs/reports/rhythm.md).
    """
    return torch.einsum("knc,kco->kno", x, w) + b.reshape(w.shape[0], 1, w.shape[2])


class _TemporalEncoder(nn.Module):
    """Depthwise-separable dilated 1D conv stack over time, weights per member.

    Operates on (K, N, Cin, T) where N is an arbitrary flattened batch
    (windows x grid cells). Shared weights across cells is enforced by the caller
    reshaping cells into N.
    """

    def __init__(self, K: int, cin: int, chans: tuple[int, ...] = (8, 12, 16),
                 kernel: int = 5, dilations: tuple[int, ...] = (1, 2, 4),
                 dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.dilations = dilations
        dims = [cin, *chans]
        self.dw = nn.ParameterList()
        self.pw = nn.ParameterList()
        self.bn = nn.ModuleList()
        for i in range(len(chans)):
            k = kernel
            self.dw.append(nn.Parameter(torch.randn(K, 1, 1, k) / k ** 0.5))
            self.pw.append(nn.Parameter(torch.randn(K, 1, dims[i], dims[i + 1]) / dims[i] ** 0.5))
            self.bn.append(nn.BatchNorm1d(dims[i + 1]))
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(K, N, Cin, T) -> (K, N, chans[-1], T)

        Two grouped convolutions, both verified numerically against a per-member
        reference loop (tests/test_rhythm_model.py):

        * depthwise: groups = K*N*Cin with a (K,1,1,k) kernel broadcast over
          cells, so every member has its own k-tap filter and every (cell,
          channel) pair is filtered independently along time.
        * pointwise: groups = K*N, one group per (member, cell) pair. The weight
          is laid out (K*N*Cout, Cin, 1) and the input (1, K*N*Cin, T), so each
          group applies that member's own Cin->Cout matrix.

        BatchNorm is per member, not per cell: with batch size 1 a per-cell
        BatchNorm would have almost no samples.
        """
        K, N, T = x.shape[0], x.shape[1], x.shape[-1]
        h = x
        cin = x.shape[2]
        for i, d in enumerate(self.dilations[: len(self.dw)]):
            k = self.dw[i].shape[-1]
            pad = (k // 2) * d
            hp = F.pad(h, (pad, pad))                                   # (K,N,cin,T+2p)
            wt = self.dw[i].reshape(K, 1, 1, k).expand(K, N, cin, k).reshape(K * N * cin, 1, k)
            h1 = F.conv1d(hp.reshape(1, K * N * cin, hp.shape[-1]), wt,
                          groups=K * N * cin, dilation=d).reshape(K, N, cin, T)
            cout = self.pw[i].shape[3]
            pw = self.pw[i].expand(K, N, cin, cout).reshape(K * N * cout, cin, 1)
            h1 = F.conv1d(h1.reshape(1, K * N * cin, T), pw,
                          groups=K * N).reshape(K, N, cout, T)
            # BatchNorm per member: batch = (cells x time) for that member, so
            # every member has its own running statistics over N*T samples.
            h1 = h1.permute(0, 2, 1, 3).reshape(K, cout, N * T)
            h1 = self.bn[i](h1).reshape(K, cout, N, T).permute(0, 2, 1, 3)
            h1 = self.drop(self.act(h1))
            h = h + h1 if cout == cin else h1
            cin = cout
        return h


class RhythmEnsemble(nn.Module):
    """K-member ensemble over grid + global channels. One usable logit per window."""

    def __init__(self, K: int = 64, grid_ch: int = 6, glob_ch: int = 8,
                 chans: tuple[int, ...] = (8, 12, 16), glob_chans: tuple[int, ...] = (8, 8),
                 kernel: int = 5, dilations: tuple[int, ...] = (1, 2, 4),
                 head_hidden: int = 32, dropout: float = 0.1, n_aux: int = 2,
                 n_cell_stats: int = 4):
        super().__init__()
        self.K = K
        self.grid_ch = grid_ch
        self.n_aux = n_aux
        self.grid_enc = _TemporalEncoder(K, grid_ch, chans, kernel, dilations, dropout)
        self.glob_enc = _TemporalEncoder(K, glob_ch, glob_chans, kernel, dilations, dropout)
        self.feat_dim = chans[-1] * 2 + glob_chans[-1] * 2 + n_cell_stats
        self.head_norm = nn.LayerNorm(self.feat_dim)
        self.head_drop = nn.Dropout(dropout)
        self.w1 = nn.Parameter(torch.randn(K, self.feat_dim, head_hidden) / self.feat_dim ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(K, head_hidden))
        self.w2 = nn.Parameter(torch.randn(K, head_hidden, 1 + n_aux) / head_hidden ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(K, 1 + n_aux))

    def forward(self, X: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        K, B, T, G, _, C = X.shape
        # ---- grid branch: fold cells into the batch; weights shared across cells
        x = X.permute(0, 1, 3, 4, 5, 2).reshape(K, B * G * G, C, T)
        h = self.grid_enc(x)                                   # (K, B*G*G, ch, T)
        ch = h.shape[2]
        h = h.reshape(K, B, G * G, ch, T)
        cell_mean = h.mean(dim=2).mean(dim=-1)                 # (K, B, ch)
        cell_max = h.max(dim=2).values.mean(dim=-1)            # (K, B, ch)
        # per-cell activity statistics: is the motion localised?
        act = h[:, :, :, -1, :].mean(dim=-1)                   # (K, B, G*G)
        stats = torch.stack([
            act.mean(dim=-1),
            act.std(dim=-1),
            act.max(dim=-1).values,
            act.max(dim=-1).values / (act.sum(dim=-1) + 1e-6),  # concentration
        ], dim=-1)                                             # (K, B, 4)
        # ---- global branch: (K,B,T,gch) -> (K,B,gch,T) for the 1D encoder
        gg = self.glob_enc(g.permute(0, 1, 3, 2))              # (K, B, gch, T)
        g_mean = gg.mean(dim=-1)
        g_max = gg.max(dim=-1).values
        feat = torch.cat([cell_mean, cell_max, g_mean, g_max, stats], dim=-1)
        feat = self.head_drop(self.head_norm(feat))
        hh = self.head_drop(F.gelu(group_linear(feat, self.w1, self.b1)))
        out = group_linear(hh, self.w2, self.b2)               # (K, B, 1+n_aux)
        return out[..., 0], out[..., 1:]


def count_params(model: RhythmEnsemble) -> dict:
    """Parameter count for ONE member (the ensemble shares nothing)."""
    per_member = sum(p.numel() for p in model.parameters()) // model.K
    return {"per_member": int(per_member),
            "ensemble_total": int(sum(p.numel() for p in model.parameters())),
            "K": model.K}
