#!/usr/bin/env python3
"""Lead tool: collect every experiment result JSON across the pod worktrees and check the registry.

The lead never trusts a chat summary — it reads the result FILES. This tool:
  1. walks each worktree's `experiments/registry.jsonl` and `eval/results/**/*.json`;
  2. prints one row per experiment id: verdict, whether a results file exists, and the headline numbers
     it can find (n, protocol, label_source, device, auroc/ci95, cost);
  3. flags registry entries marked `done` that have NO results file (honesty check);
  4. flags result files that look non-empty but carry no `n` (incomplete).

usage: python3 tools/pod/collect_results.py --wt /workspace/wt/exp-mosaic /workspace/wt/exp-fps-pod ...
       python3 tools/pod/collect_results.py --wt /workspace/wt/* --json-out /workspace/state/wave_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

NUMKEYS = ("auroc", "auroc_ci95", "ci95", "bal_acc", "accuracy", "n", "n_bad", "n_good")


def find_registry(wt: str) -> str | None:
    p = os.path.join(wt, "experiments", "registry.jsonl")
    return p if os.path.exists(p) else None


def load_registry(path: str) -> dict[str, dict]:
    reg: dict[str, dict] = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in e:
            reg.setdefault(e["id"], {}).update(e)
    return reg


def flatten_numbers(obj, prefix="", depth=0, out=None):
    if out is None:
        out = {}
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in NUMKEYS and isinstance(v, (int, float, list)):
                out[prefix + k] = v
            elif isinstance(v, (dict, list)):
                flatten_numbers(v, prefix + k + ".", depth + 1, out)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for i, v in enumerate(obj[:4]):
            flatten_numbers(v, prefix + f"[{i}].", depth + 1, out)
    return out


def result_files(wt: str):
    pats = [
        os.path.join(wt, "eval", "results", "**", "*.json"),
        os.path.join(wt, "eval", "results", "*.json"),
    ]
    seen = {}
    for p in pats:
        for f in glob.glob(p, recursive=True):
            seen[os.path.relpath(f, wt)] = f
    return seen


def bundle_lookup(rf: str, bundle: str) -> str | None:
    """The historical (laptop) registry points at absolute laptop paths. The same JSON files
    are shipped inside the data bundle; find them by basename under <bundle>/local_work."""
    if not rf or not bundle:
        return None
    base = os.path.basename(rf)
    for root in (os.path.join(bundle, "local_work"), os.path.join(bundle, "experiments_local"), bundle):
        for p in glob.glob(os.path.join(root, "**", base), recursive=True):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt", nargs="+", required=True)
    ap.add_argument("--bundle", default=os.environ.get("JADAL_BUNDLE", "/workspace/jadal_pod_bundle"))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    rows = []
    problems = []
    for wt in a.wt:
        if not os.path.isdir(wt):
            continue
        name = os.path.basename(wt.rstrip("/"))
        regp = find_registry(wt)
        reg = load_registry(regp) if regp else {}
        files = result_files(wt)
        for eid, e in sorted(reg.items()):
            rf = e.get("results")
            # resolve the results path relative to the worktree if relative
            cand = None
            if rf:
                cand = rf if os.path.isabs(rf) and os.path.exists(rf) else os.path.join(wt, rf.lstrip("/"))
                if isinstance(rf, str) and rf.startswith("/workspace/") and not os.path.exists(rf):
                    cand = os.path.join(wt, rf.replace("/workspace/", "", 1))
            exists = bool(cand and os.path.exists(cand))
            nums = {}
            if exists:
                try:
                    d = json.load(open(cand))
                    nums = flatten_numbers(d)
                except Exception as ex:  # noqa: BLE001
                    problems.append(f"{name}:{eid}: unreadable results file {cand} ({ex})")
            elif rf:
                found = bundle_lookup(rf, a.bundle)
                if found:
                    cand = found
                    exists = True
                    try:
                        nums = flatten_numbers(json.load(open(found)))
                    except Exception:  # noqa: BLE001
                        pass
                    problems.append(f"{name}:{eid}: results file relocated to the bundle: {found}")
                else:
                    problems.append(f"{name}:{eid}: status={e.get('status')} but NO results file ({rf!r})")
            elif e.get("status") == "done":
                problems.append(f"{name}:{eid}: status=done but no results path recorded")
            if exists and not nums:
                problems.append(f"{name}:{eid}: results file has no recognised numbers (n/auroc/...)")
            rows.append(
                {
                    "worktree": name,
                    "id": eid,
                    "track": e.get("track"),
                    "status": e.get("status"),
                    "verdict": e.get("verdict"),
                    "title": (e.get("title") or "")[:60],
                    "results_file": (
                        os.path.relpath(cand, wt) if cand and os.path.exists(cand) and str(cand).startswith(wt)
                        else (cand if cand and os.path.exists(cand) else None)
                    ),                    "numbers": nums,
                    "summary": (e.get("summary") or "")[:200],
                }
            )

    for r in rows:
        n = r["numbers"].get("n", "?")
        au = r["numbers"].get("auroc", r["numbers"].get("auroc_ci95", "-"))
        print(
            f"{r['worktree']:16s} {r['id']:8s} {str(r['status']):7s} {str(r['verdict']):12s} "
            f"n={n} auroc={au} {'OK' if r['results_file'] else 'NO-FILE'}"
        )
    print(f"\n{len(rows)} registry entries across {len(a.wt)} worktrees")
    if problems:
        print("\nPROBLEMS (honesty check):")
        for p in problems:
            print("  -", p)
    else:
        print("no problems")
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        json.dump(rows, open(a.json_out, "w"), indent=1)
        print("wrote", a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
