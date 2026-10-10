"""Batched training: one BatchedHeads module per hidden-size group, single Adam.

Deterministic: torch.manual_seed(seed), fixed arange batch order, float32,
no shuffle/dropout. Full-batch Adam (N<=10k, D<=821, K<=256 -> tiny).
Returns per-config logits history + final state.
"""
from __future__ import annotations

import time

import torch

from .heads import BatchedHeads


def train_groups(X, y, groups: dict[int, list], d_in: int, epochs=50, batch=2048,
                 device="cuda", seed=0, verbose=False):
    torch.manual_seed(seed)
    X = X.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    n = X.shape[0]
    mods, opts, sched = {}, {}, {}
    pos_frac = float(y.mean().item())
    for h, cfgs in groups.items():
        m = BatchedHeads(d_in, h, len(cfgs), seed=seed).to(device, torch.float32)
        mods[h] = m
        # per-head LR/WD differ -> use per-group max lr, per-head wd via manual scaling:
        # simpler: one Adam per group with lr = max lr in group; wd handled per-head below.
        lr = max(c.lr for c in cfgs)
        opts[h] = torch.optim.Adam(m.parameters(), lr=lr)
    # pos_weight per head (class_weight flag)
    pw = {}
    for h, cfgs in groups.items():
        w = torch.ones(len(cfgs), device=device)
        for i, c in enumerate(cfgs):
            if c.class_weight and 0 < pos_frac < 1:
                w[i] = (1 - pos_frac) / pos_frac
        pw[h] = w
    order = torch.arange(n, device=device)  # fixed order
    t0 = time.time()
    for ep in range(epochs):
        for h, m in mods.items():
            opts[h].zero_grad()
            loss_sum = 0.0
            cfgs = groups[h]
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                xb, yb = X[idx], y[idx]
                logits = m(xb)  # (K,B)
                yb_e = yb.unsqueeze(0).expand(len(cfgs), -1)
                pw_e = pw[h].unsqueeze(1).expand(-1, yb.numel())
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb_e, pos_weight=pw_e, reduction="none").mean(dim=1)
                # per-head weight decay (manual, since Adam wd is global)
                wd_term = torch.zeros(len(cfgs), device=device)
                for i, c in enumerate(cfgs):
                    if c.wd > 0:
                        sq = sum((p[i] ** 2).sum() for p in m.parameters())
                        wd_term[i] = c.wd * sq
                loss = (bce + wd_term).sum()
                loss = loss * (idx.numel() / n)
                loss_sum = loss_sum + loss
            loss_sum.backward()
            opts[h].step()
    dt = time.time() - t0
    # final logits on train
    out = {}
    with torch.no_grad():
        for h, m in mods.items():
            out[h] = m(X).detach()
    return mods, out, dt
