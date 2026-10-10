"""Head configs: hidden size {0 (linear), 64, 256, 1024} x wd x seed x class-weight."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadConfig:
    hidden: int  # 0 = linear
    wd: float
    lr: float
    seed: int
    class_weight: bool  # pos_weight = neg/pos
    dropout: float = 0.0  # always 0: dropout-free per spec


def build_grid(seeds=(0, 1), lrs=(0.05,), wds=(0.0, 1e-4, 1e-3),
               hiddens=(0, 64, 256, 1024), class_weights=(False, True)):
    cfgs = []
    for h in hiddens:
        for wd in wds:
            for lr in lrs:
                for s in seeds:
                    for cw in class_weights:
                        cfgs.append(HeadConfig(hidden=h, wd=wd, lr=lr, seed=s,
                                               class_weight=cw))
    return cfgs
