#!/usr/bin/env python3
"""Union several append-only experiment registries and render one EXPERIMENTS.md.

The pod runs one worktree (one branch) per concern; each branch carries its own
`experiments/registry.jsonl`. This tool merges them the same way `tools/exp.py`
resolves its own registry: latest line per id wins, ids sorted.

usage:
  union_registry.py --in a/experiments/registry.jsonl b/experiments/registry.jsonl \
                    [--json-out experiments/registry.jsonl] [--md-out experiments/EXPERIMENTS.md]
  union_registry.py --in a.jsonl b.jsonl c.jsonl --report     # print a summary table only
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def load(paths):
    reg: dict[str, dict] = {}
    per_file = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"union_registry: missing {p}", file=sys.stderr)
            continue
        n = 0
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                print(f"union_registry: bad JSON in {p}: {line[:80]}", file=sys.stderr)
                continue
            if "id" not in e:
                continue
            e.setdefault("_sources", [])
            e["_sources"] = list(reg.get(e["id"], {}).get("_sources", [])) + [p]
            reg.setdefault(e["id"], {}).update(e)
            n += 1
        per_file[p] = n
    return reg, per_file


def render_md(reg: dict[str, dict]) -> str:
    out = [
        "# Experiment log",
        "",
        "Every experiment: why it was run, on what data, what came out, and the decision. "
        "Generated from the union of the pod worktrees' `experiments/registry.jsonl` files "
        "(append-only; the latest line per id wins). Negative results are kept.",
        "",
        "| id | track | status | verdict | title | data / regime |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in sorted(reg.items()):
        out.append(
            f"| {i} | {e.get('track','')} | {e.get('status','')} | {e.get('verdict','-')} | "
            f"{e.get('title','')} | {str(e.get('data',''))[:60]} / {e.get('regime','') or ''} |"
        )
    out.append("")
    for i, e in sorted(reg.items()):
        out += [
            f"## {i} — {e.get('title','')}",
            f"- **Why:** {e.get('why','')}",
            f"- **Hypothesis:** {e.get('hypothesis','')}",
            f"- **Data / label source:** {e.get('data','')}",
            f"- **Regime:** {e.get('regime','n/a')}",
            f"- **Method:** {e.get('method','')}",
            f"- **Owner / started / finished:** {e.get('owner','')} / {e.get('started','')} / {e.get('finished','—')}",
            f"- **Verdict:** {e.get('verdict','pending')} — {e.get('summary','')}",
            f"- **Decision:** {e.get('decision','')}",
            f"- **Cost:** {e.get('cost','')}",
            f"- **Results:** {e.get('results','')}",
            "",
        ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    reg, per_file = load(a.inputs)
    for p, n in per_file.items():
        print(f"  {n:4d} lines  {p}")
    done = [e for e in reg.values() if e.get("status") == "done"]
    print(f"union: {len(reg)} ids ({len(done)} done) from {len(per_file)} files")
    if a.report:
        for e in sorted(reg.values(), key=lambda x: x["id"]):
            print(f"{e['id']:8s} {e.get('status','?'):8s} {e.get('verdict','-'):13s} {e.get('title','')[:70]}")
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            for e in sorted(reg.values(), key=lambda x: x["id"]):
                e.pop("_sources", None)
                f.write(json.dumps(e) + "\n")
        print("wrote", a.json_out)
    if a.md_out:
        os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
        with open(a.md_out, "w") as f:
            f.write(render_md(reg))
        print("wrote", a.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
