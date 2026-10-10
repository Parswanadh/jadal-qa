"""Vectorised K-head MLP trained simultaneously with torch.bmm.

Shapes: X (N,D) float32. Params stacked over K:
  linear: W (K,D,1), b (K,)
  mlp:    W1 (K,H,D), b1 (K,H), W2 (K,1,H), b2 (K,)
Forward broadcasts X -> (K,N,D) via expand (no copy) then bmm.
Deterministic: fixed batch order (arange slices), no dropout/shuffle.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BatchedHeads(nn.Module):
    def __init__(self, d_in: int, hidden: int, k: int, seed: int = 0):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.k = k
        g = torch.Generator().manual_seed(seed)
        if hidden <= 0:
            w = torch.randn(k, d_in, 1, generator=g) * (1.0 / max(1, d_in) ** 0.5)
            self.W = nn.Parameter(w)
            self.b = nn.Parameter(torch.zeros(k))
            self.W1 = None
        else:
            w1 = torch.randn(k, hidden, d_in, generator=g) * (2.0 / (d_in + hidden)) ** 0.5
            self.W1 = nn.Parameter(w1)
            self.b1 = nn.Parameter(torch.zeros(k, hidden))
            w2 = torch.randn(k, 1, hidden, generator=g) * (2.0 / (hidden + 1)) ** 0.5
            self.W2 = nn.Parameter(w2)
            self.b2 = nn.Parameter(torch.zeros(k))
            self.W = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,D) -> logits (K,N)
        n = x.shape[0]
        xe = x.unsqueeze(0).expand(self.k, n, self.d_in)
        if self.hidden <= 0:
            out = torch.bmm(xe, self.W).squeeze(-1) + self.b.unsqueeze(1)
            return out
        h = torch.bmm(xe, self.W1.transpose(1, 2)) + self.b1.unsqueeze(1)
        h = torch.relu(h)
        out = torch.bmm(h, self.W2.transpose(1, 2)).squeeze(-1) + self.b2.unsqueeze(1)
        return out
