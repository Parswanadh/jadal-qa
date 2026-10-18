"""VIA temporal-segment parsing for the IndEgo keystep annotations.

Shared with stream S5 (`dsh/task`, TASK.00b): the observed structure is
VGG Image Annotator JSON where ``attribute["1"].aname`` names the attribute,
``attribute["1"].options`` maps option-id -> step label, ``file["1"].fname`` is the
video file, and each ``metadata[*]`` has ``z = [start_s, end_s]`` in SECONDS and
``av["1"]`` = the chosen option-id.  Verified against real clip durations there.

Duplicated here (rather than imported across branches) so the V-track PR is
self-contained; behaviour is identical.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def parse_via(path: str):
    with open(path) as fh:
        d = json.load(fh)
    attrs = d.get("attribute", {})
    fid = sorted(d.get("file", {}).keys())
    if not attrs or not fid:
        return None
    a = attrs[sorted(attrs.keys())[0]]
    opts = {str(k): str(v) for k, v in (a.get("options") or {}).items()}
    fname = d["file"][fid[0]].get("fname")
    segs = []
    for _, m in (d.get("metadata") or {}).items():
        z = m.get("z")
        if not z or len(z) < 2:
            continue
        av = m.get("av") or {}
        opt = av.get(sorted(av.keys())[0]) if av else None
        segs.append({"start_s": float(z[0]), "end_s": float(z[1]),
                     "option_id": str(opt) if opt is not None else None,
                     "step": opts.get(str(opt), None)})
    segs.sort(key=lambda s: s["start_s"])
    return {"aname": a.get("aname"), "fname": fname, "options": opts,
            "segments": segs, "path": path}


def find_annotations(root="/home/parshu/projects/awing/data/indego"):
    """{fname: annotation_path} for every VIA file under the IndEgo tree."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "AS_*.json"), recursive=True):
        try:
            v = parse_via(p)
        except Exception:
            continue
        if v and v.get("fname"):
            out.setdefault(v["fname"], p)
    return out


def gap_list(segments, duration_s=None):
    """Un-annotated stretches between the first and last annotated step."""
    gaps, prev_end = [], None
    for s in segments:
        if prev_end is None:
            gaps.append({"start_s": 0.0, "end_s": round(s["start_s"], 3)})
        elif s["start_s"] - prev_end > 1e-9:
            gaps.append({"start_s": round(prev_end, 3), "end_s": round(s["start_s"], 3)})
        prev_end = s["end_s"]
    if duration_s and prev_end is not None and duration_s - prev_end > 1e-9:
        gaps.append({"start_s": round(prev_end, 3), "end_s": round(float(duration_s), 3)})
    return [g for g in gaps if g["end_s"] - g["start_s"] > 1e-9]


def l1_truth(segments, t0: float, t1: float) -> str:
    """``advancing`` if the slice is inside an annotated step, else ``no_relevant_work``.

    A slice is ``advancing`` when it overlaps a step for at least half of its length;
    a slice in a gap is ``no_relevant_work``.  This is the free L1 ground truth; it is
    IndEgo demo-style assembly and may not transfer to factory work.
    """
    if not segments:
        return "no_relevant_work"
    cover = 0.0
    for s in segments:
        cover += max(0.0, min(t1, s["end_s"]) - max(t0, s["start_s"]))
    return "advancing" if cover >= 0.5 * (t1 - t0) else "no_relevant_work"


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float((a.sum() / n) * (b.sum() / n) + ((n - a.sum()) / n) * ((n - b.sum()) / n))
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > 1e-12 else float("nan")


def slice_starts(duration_s: float, n_slices: int = 6, slice_s: float = 30.0):
    """Evenly spaced slice starts spanning the clip, including the final slice
    (the end state matters for completion)."""
    if duration_s <= slice_s:
        return [0.0]
    return [round(x, 2) for x in np.linspace(0.0, duration_s - slice_s, n_slices)]
