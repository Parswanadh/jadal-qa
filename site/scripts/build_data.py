"""Bake the explorer's data from files in the repo.

Run from site/:  python3 -I scripts/build_data.py

Writes src/data/bake.json (figures + funnel + efficiency) and
src/data/registry.json (experiment log rows). Every figure carries a
`source` block (repo-relative file, JSON key path or literal, and line) so
src/test/traces.test.ts can re-check it against the file.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
REPO = os.path.dirname(SITE)
OUT = os.path.join(SITE, "src", "data")


def rel(path):
    return os.path.relpath(path, REPO).replace(os.sep, "/")


def read(path):
    with open(os.path.join(REPO, path), encoding="utf-8") as fh:
        return fh.read()


def line_of(path, needle):
    for i, line in enumerate(read(path).splitlines(), 1):
        if needle in line:
            return i
    raise SystemExit(f"literal not found in {path}: {needle!r}")


def jget(path, keys):
    node = json.loads(read(path))
    for k in keys:
        node = node[k]
    return node


def json_fig(fid, label, display, path, keys):
    value = jget(path, keys)
    return {
        "id": fid,
        "label": label,
        "value": value,
        "display": display,
        "source": {"file": path, "keys": keys, "line": None},
    }


def literal_fig(fid, label, display, value, path, literal):
    return {
        "id": fid,
        "label": label,
        "value": value,
        "display": display,
        "source": {"file": path, "literal": literal, "line": line_of(path, literal)},
    }


RUN = "eval/results/poc/run_summary.json"
NESTED = "eval/results/ensemble/nested_loco.json"
SYNC = "docs/pod/SYNC_UPDATE_2026-10-10_2145.md"
README = "README.md"
CASCADE = "jadal/poc/cascade.py"
REG = "experiments/registry.jsonl"
E101 = "eval/results/exp_fps/e1_01.json"


def registry_rows():
    """Latest line per id wins. The id-count check runs on the raw lines before
    the dedupe, so it can fail if the dedupe ever drops or invents ids."""
    raw_ids = []
    records = []
    with open(os.path.join(REPO, REG), encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            rec = json.loads(raw)
            raw_ids.append(rec["id"])
            records.append((lineno, rec))
    distinct = len(set(raw_ids))
    latest = {}
    order = []
    for lineno, rec in records:
        rid = rec["id"]
        if rid not in latest:
            order.append(rid)
        latest[rid] = (lineno, rec)
    if len(order) != distinct:
        raise SystemExit(f"registry dedupe dropped ids: {distinct} distinct raw, {len(order)} kept")
    rows = []
    for rid in order:
        lineno, rec = latest[rid]
        rows.append({
            "id": rid,
            "track": rec.get("track") or rid.split(".")[0],
            "title": rec.get("title") or rid,
            "status": rec.get("status") or "unknown",
            "verdict": rec.get("verdict"),
            "summary": rec.get("summary") or "",
            "results": rec.get("results"),
            "finished": rec.get("finished"),
            "line": lineno,
        })
    return rows


def s0_rules():
    src = read(CASCADE)
    rules = []
    for m in re.finditer(r'\("(\w+)", "(\w+)", "([<>])", ([0-9.e-]+)\)', src):
        code, key, op, thr = m.group(1), m.group(2), m.group(3), m.group(4)
        rules.append({
            "code": code,
            "feature": key,
            "op": op,
            "threshold": float(thr),
            "source": {"file": CASCADE, "literal": m.group(0),
                       "line": line_of(CASCADE, m.group(0))},
        })
    if len(rules) != 5:
        raise SystemExit(f"expected 5 S0 rules, found {len(rules)}")
    return rules


def reject_record():
    text = read(README)
    m = re.search(r"```json\n(\{\n  \"window_id\".*?\n\})\n```", text, re.S)
    if not m:
        raise SystemExit("reject record block not found in README")
    record = json.loads(m.group(1))
    return {
        "record": record,
        "source": {"file": README, "literal": '"window_id": "f069_w004_c00077_t0000"',
                   "line": line_of(README, '"window_id": "f069_w004_c00077_t0000"')},
    }


def e101():
    d = json.loads(read(E101))
    rows = []
    for r in d["rows"]:
        rows.append({
            "fps": r["fps"],
            "frames": r["frames_per_window"],
            "auroc": r["auroc"],
            "ci": r["ci95"],
            "auroc_spatial": r["auroc_spatial"],
            "ci_spatial": r["ci95_spatial"],
            "feat_ms": r["feat_ms_mean"],
        })
    return {
        "n": d["n"], "n_good": d["n_good"], "n_bad": d["n_bad"],
        "label_source": d["label_source"],
        "claim_status": d["claim_status"],
        "rows": rows,
        "source": {"file": E101, "keys": ["rows"], "line": None},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    s0 = s0_rules()
    figures = [
        json_fig("windows", "windows scored (PoC run)", "1,587", RUN, ["unique_windows"]),
        json_fig("hours", "video hours", "4.404 h", RUN, ["hours"]),
        json_fig("factories", "factories in the PoC run", "47", RUN, ["factories"]),
        json_fig("accept", "accept", "709", RUN, ["verdicts", "accept"]),
        json_fig("reject", "reject", "702", RUN, ["verdicts", "reject"]),
        json_fig("review", "review", "176", RUN, ["verdicts", "review"]),
        json_fig("reject_share", "reject share", "44.2%", RUN, ["verdict_share", "reject"]),
        json_fig("s0", "decided at S0", "0", RUN, ["stages", "S0"]),
        json_fig("s1", "decided at S1", "1,411", RUN, ["stages", "S1"]),
        json_fig("s2", "sent to S2 review", "176", RUN, ["stages", "S2"]),
        json_fig("indomain_auroc", "in-domain ensemble AUROC (IndEgo, n=150)", "0.834",
                 NESTED, ["views", "ensemble_probavg", "auroc"]),
        json_fig("indomain_ci_lo", "in-domain CI low", "0.715", NESTED,
                 ["views", "ensemble_probavg", "ci95", 0]),
        json_fig("indomain_ci_hi", "in-domain CI high", "0.914", NESTED,
                 ["views", "ensemble_probavg", "ci95", 1]),
        json_fig("gate_auroc", "pre-registered gate AUROC >= 0.85 met", "no", NESTED,
                 ["gates", "AUROC>=0.85"]),
        json_fig("gate_ci", "pre-registered gate CI-low >= 0.72 met", "no", NESTED,
                 ["gates", "CI_low>=0.72"]),
        literal_fig("ood_ensemble", "frozen ensemble, factory footage", "0.566 [0.433, 0.693]",
                    0.566, SYNC, "0.566 (0.433-0.693)"),
        literal_fig("ood_cascade", "PoC cascade, factory footage", "0.600 [0.474, 0.724]",
                    0.600, SYNC, "0.600 (0.474-0.724)"),
        literal_fig("ood_rejects", "good windows rejected by the cascade", "53 of 138",
                    53, SYNC, "rejects 53 of 138 good windows (38%)"),
        literal_fig("ood_bad_rejected", "bad windows rejected by the cascade", "13 of 22",
                    13, SYNC, "13 of 22 bad"),
        literal_fig("ood_ens_lo", "frozen ensemble CI low", "0.433", 0.433, SYNC,
                    "0.566 (0.433-0.693)"),
        literal_fig("ood_ens_hi", "frozen ensemble CI high", "0.693", 0.693, SYNC,
                    "0.566 (0.433-0.693)"),
        literal_fig("ood_casc_lo", "PoC cascade CI low", "0.474", 0.474, SYNC,
                    "0.600 (0.474-0.724)"),
        literal_fig("ood_casc_hi", "PoC cascade CI high", "0.724", 0.724, SYNC,
                    "0.600 (0.474-0.724)"),
        json_fig("indomain_n", "in-domain windows (IndEgo, LOCO)", "150", NESTED, ["n"]),
        json_fig("e101_features", "per-pixel features in the S1 view", "26", E101,
                 ["rows", 0, "n_features"]),
        literal_fig("s1_legacy_threshold", "legacy threshold field in S1 receipts (never decides a verdict)", "0.6", 0.6, CASCADE,
                    "THRESHOLD = 0.6"),
        literal_fig("s1_margin", "S2 review margin", "0.25", 0.25, CASCADE,
                    "MARGIN = 0.25"),
        literal_fig("ood_nbad", "labelled bad windows (factory, n_bad)", "22", 22, SYNC,
                    "160 random Egocentric-10K windows, 138 good / 22 bad"),
    ]
    reg = registry_rows()
    reg_lines = sum(1 for raw in read(REG).splitlines() if raw.strip())

    bake = {
        "figures": figures,
        "s0_rules": s0,
        "reject": reject_record(),
        "e101": e101(),
        "res03": next(r for r in reg if r["id"] == "RES.03"),
        "registry_lines": reg_lines,
        "registry_unique": len(reg),
    }
    with open(os.path.join(OUT, "bake.json"), "w", encoding="utf-8") as fh:
        json.dump(bake, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(OUT, "registry.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": REG, "rows": reg}, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {len(figures)} figures, {len(reg)} registry rows")


if __name__ == "__main__":
    main()
