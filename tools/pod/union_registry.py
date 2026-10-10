#!/usr/bin/env python3
"""Union several experiment registry files and render experiments/EXPERIMENTS.md.

Merge rule (identical to `tools/exp.py`): the registry is append-only and the LAST
line seen for an experiment id wins, with partial lines merged into the earlier entry.
Across several sources the sources listed LAST win for overlapping ids, so the live
registry is listed after the historical snapshot.

Usage
  python3 tools/pod/union_registry.py                       # merge defaults + render EXPERIMENTS.md
  python3 tools/pod/union_registry.py --source a.jsonl --source b.jsonl
  python3 tools/pod/union_registry.py --merged out.jsonl    # also write the merged registry
  python3 tools/pod/union_registry.py --seed                # append snapshot lines missing from registry.jsonl

Defaults:
  sources        experiments/registry_snapshot.jsonl  experiments/registry.jsonl
  out            experiments/EXPERIMENTS.md
  seed target    experiments/registry.jsonl

Rendering is delegated to `python3 tools/exp.py render` (JADAL_EXP_DIR pointed at a
temporary directory holding the merged registry.jsonl), so EXPERIMENTS.md is always
the file exp.py would produce. --seed is append-only: it only adds lines whose id is
absent from registry.jsonl and never rewrites existing lines, so a re-run is a no-op.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SOURCES = [
    os.path.join(REPO, "experiments", "registry_snapshot.jsonl"),
    os.path.join(REPO, "experiments", "registry.jsonl"),
]


def read_lines(path):
    """Return the parsed JSON entries of one registry file, in file order."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit("%s:%d: not valid JSON: %s" % (path, lineno, exc))
    return out


def merge(sources):
    """Merge ordered sources into {id: entry}; later lines/sources win (exp.py rule)."""
    merged = {}
    for path in sources:
        for entry in read_lines(path):
            if "id" not in entry:
                raise SystemExit("%s: registry line without an 'id': %r" % (path, entry))
            merged.setdefault(entry["id"], {}).update(entry)
    return merged


def seed(target, sources):
    """Append, to `target` (append-only), EVERY line of the ids that target does not know yet.

    The whole run of lines for a new id is copied, in order, so that the
    "latest line per id wins" rule keeps holding after the merge (copying only the
    first line of an id would re-open the entry from an older, partial line).
    The id set is fixed before appending, so a re-run adds nothing.
    """
    have = set(e["id"] for e in read_lines(target))
    added = []
    with open(target, "a") as fh:
        for path in sources:
            for entry in read_lines(path):
                if entry["id"] in have:
                    continue
                fh.write(json.dumps(entry) + "\n")
                added.append(entry["id"])
    return added


def render(merged, out_path, exp_py=None):
    """Write EXPERIMENTS.md by calling tools/exp.py render on the merged registry."""
    exp_py = exp_py or os.path.join(REPO, "tools", "exp.py")
    with tempfile.TemporaryDirectory(prefix="jadal-registry-") as tmp:
        with open(os.path.join(tmp, "registry.jsonl"), "w") as fh:
            for eid in sorted(merged):
                fh.write(json.dumps(merged[eid]) + "\n")
        env = dict(os.environ, JADAL_EXP_DIR=tmp)
        subprocess.run([sys.executable, exp_py, "render"], env=env, check=True,
                       stdout=subprocess.DEVNULL)
        with open(os.path.join(tmp, "EXPERIMENTS.md")) as fh:
            text = fh.read()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", default=None,
                    help="registry jsonl (repeatable, later source wins); default: snapshot then live registry")
    ap.add_argument("--out", default=os.path.join(REPO, "experiments", "EXPERIMENTS.md"),
                    help="rendered markdown path (default: experiments/EXPERIMENTS.md)")
    ap.add_argument("--merged", default=None, help="also write the merged registry to this jsonl path")
    ap.add_argument("--seed", action="store_true",
                    help="append snapshot lines whose id is missing from the live registry (append-only)")
    ap.add_argument("--seed-target", default=os.path.join(REPO, "experiments", "registry.jsonl"))
    ap.add_argument("--seed-source", action="append", default=None,
                    help="source used by --seed (repeatable); default: every --source except the live registry")
    a = ap.parse_args(argv)

    sources = a.source or DEFAULT_SOURCES
    for p in sources:
        if not os.path.exists(p):
            print("union_registry: warning: source not found: %s" % p, file=sys.stderr)

    if a.seed:
        seed_sources = a.seed_source or [p for p in sources if os.path.abspath(p) != os.path.abspath(a.seed_target)]
        added = seed(a.seed_target, seed_sources)
        print("seed: appended %d line(s) to %s%s" % (len(added), a.seed_target,
              (" (%s)" % ", ".join(sorted(set(added)))) if added else ""))

    merged = merge(sources)
    if a.merged:
        with open(a.merged, "w") as fh:
            for eid in sorted(merged):
                fh.write(json.dumps(merged[eid]) + "\n")
        print("merged registry -> %s (%d experiments)" % (a.merged, len(merged)))
    render(merged, a.out)
    print("rendered %d experiments -> %s" % (len(merged), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
