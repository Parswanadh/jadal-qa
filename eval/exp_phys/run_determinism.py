"""Q3 determinism proof: run the deterministic rule scorer 10x on a fixed set.

Asserts, per window, that the receipt SHA-256 is identical across all 10 runs,
and additionally that the receipt *bytes* are identical. Cross-checks:
  * canonical (sorted-key) JSON is insensitive to feature dict insertion order;
  * no time/random/device keys and no floats anywhere in a receipt;
  * no GPU/CUDA/torch in the scorer's process (CPU only);
  * cross-process determinism under different PYTHONHASHSEED values.

Writes eval/results/exp_phys/Q3.01.json (the 10xN hash matrix).
Run: OMP_NUM_THREADS=2 python3 eval/exp_phys/run_determinism.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from eval.exp_phys import det_rule_scorer as DRS  # noqa: E402

INPUT = os.path.join(HERE, "input_windows.json")
RULES = os.path.join(HERE, "rule_table.json")
OUT = os.path.join(REPO, "eval", "results", "exp_phys", "Q3.01.json")
N_RUNS = 10
HASHSEEDS = ("0", "1", "42")


def load_input():
    with open(INPUT, "r", encoding="utf-8") as fh:
        return json.load(fh, parse_float=Decimal)


def run_once(windows, table, code_hash):
    """One full pass -> {window_id: (receipt, canonical_body_bytes)}."""
    out = {}
    for w in windows:
        receipt = DRS.score_window(w, table, code_hash=code_hash)
        body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        out[w["window_id"]] = (receipt, DRS.canonical_json(body))
    return out


def subprocess_digest(windows, hashseed):
    """Re-run the scorer in a fresh interpreter with a fixed PYTHONHASHSEED."""
    script = r"""
import hashlib, json, os, sys
sys.path.insert(0, %r); sys.path.insert(0, %r)
from decimal import Decimal
from eval.exp_phys import det_rule_scorer as D
with open(%r) as fh: inp = json.load(fh, parse_float=Decimal)
with open(%r) as fh: tab = json.load(fh)
ch = D.code_sha256()
h = hashlib.sha256()
for w in inp["windows"]:
    r = D.score_window(w, tab, code_hash=ch)
    h.update(r["receipt_sha256"].encode())
print(h.hexdigest())
""" % (REPO, HERE, INPUT, RULES)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["OMP_NUM_THREADS"] = "2"
    p = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, env=env, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"subprocess failed: {p.stderr[-500:]}")
    return p.stdout.strip()


def main() -> None:
    data = load_input()
    windows = data["windows"]
    table = DRS.load_rule_table(RULES)
    code_hash = DRS.code_sha256()
    rtab_hash = DRS.rule_table_sha256(table)

    # ---- 10 in-process runs -------------------------------------------------
    per_run = []
    for _ in range(N_RUNS):
        per_run.append(run_once(windows, table, code_hash))

    ids = [w["window_id"] for w in windows]
    hash_matrix = {wid: [per_run[r][wid][0]["receipt_sha256"]
                         for r in range(N_RUNS)] for wid in ids}
    body_matrix = {wid: [per_run[r][wid][1] for r in range(N_RUNS)]
                   for wid in ids}

    unique_per_window = {wid: len(set(hash_matrix[wid])) for wid in ids}
    n_identical = sum(1 for wid in ids if unique_per_window[wid] == 1)
    all_identical = n_identical == len(ids)

    # byte-identical bodies across runs (stronger than hash equality)
    bodies_identical = all(len(set(body_matrix[wid])) == 1 for wid in ids)

    # ---- receipt invariants (run 0) -----------------------------------------
    sample_receipt = per_run[0][ids[0]][0]
    forbidden_ok = True
    for wid in ids:
        try:
            DRS.assert_no_forbidden_fields(per_run[0][wid][0])
        except ValueError:
            forbidden_ok = False
            break

    # ---- canonicalisation: feature insertion order must not matter ----------
    order_ok = True
    for w in windows[:40]:
        rev = dict(reversed(list(w["features"].items())))
        w2 = dict(w)
        w2["features"] = rev
        a = DRS.score_window(w, table, code_hash=code_hash)["receipt_sha256"]
        b = DRS.score_window(w2, table, code_hash=code_hash)["receipt_sha256"]
        if a != b:
            order_ok = False
            break

    # ---- CPU only: no torch / cuda in this process --------------------------
    gpu_modules = [m for m in ("torch", "cupy", "tensorflow")
                   if m in sys.modules]

    # ---- cross-process determinism under hash randomisation -----------------
    sub_digests = [subprocess_digest(windows, hs) for hs in HASHSEEDS]
    subprocess_ok = len(set(sub_digests)) == 1

    # ---- decision distribution ----------------------------------------------
    dist = Counter(per_run[0][wid][0]["decision"].split(":")[0] for wid in ids)
    dist_by_split = {}
    for w in windows:
        d = per_run[0][w["window_id"]][0]["decision"].split(":")[0]
        dist_by_split.setdefault(w["split"], Counter())[d] += 1

    result = {
        "experiment": "Q3.01",
        "title": "Byte-identical receipt proof for the deterministic rule path",
        "claim": ("On a fixed input set of real windows (150 gold IndEgo + 30 OOD "
                  "Egocentric-10K), the deterministic rule scorer produces the "
                  "same receipt SHA-256 for every window across 10 runs."),
        "status": "PASS" if (all_identical and bodies_identical
                             and forbidden_ok and order_ok
                             and not gpu_modules and subprocess_ok) else "FAIL",
        "n_windows": len(ids),
        "n_gold": data["n_gold"],
        "n_ood": data["n_ood"],
        "n_runs": N_RUNS,
        "all_identical": bool(all_identical),
        "n_windows_identical": int(n_identical),
        "bodies_byte_identical": bool(bodies_identical),
        "receipt_sha256": {wid: hash_matrix[wid][0] for wid in ids},
        "hash_matrix": hash_matrix,
        "unique_hashes_per_window": unique_per_window,
        "receipt_schema": {
            "body_fields": ["window_id", "decision", "evidence", "feature_digest",
                            "rule_table_sha256", "code_sha256", "q"],
            "envelope_fields": ["receipt_sha256"],
            "evidence_entry": ["rule", "feature", "value", "op", "threshold"],
            "hash": "sha256(canonical_json(body))",
        },
        "canonicalisation": {
            "json": "json.dumps(obj, sort_keys=True, separators=(',',':'), "
                    "ensure_ascii=False).encode('utf-8')",
            "quantisation": "Q=4 fixed point: round(value * 10**4), "
                            "decimal ROUND_HALF_EVEN; comparisons on ints only",
            "no_floats_in_body": True,
            "no_time_random_device_fields": True,
        },
        "rule_table_sha256": rtab_hash,
        "code_sha256": code_hash,
        "checks": {
            "all_10_receipt_hashes_identical": bool(all_identical),
            "receipt_bytes_identical_across_runs": bool(bodies_identical),
            "sorted_keys_immune_to_feature_order": bool(order_ok),
            "no_forbidden_fields_no_floats": bool(forbidden_ok),
            "cpu_only_no_gpu_modules": not gpu_modules,
            "gpu_modules_seen": gpu_modules,
            "cross_process_pythonhashseed_deterministic": bool(subprocess_ok),
            "pythonhashseed_digests": sub_digests,
        },
        "decision_distribution": {
            "overall": dict(dist),
            "by_split": {k: dict(v) for k, v in dist_by_split.items()},
        },
        "sample_receipt": sample_receipt,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "device": "CPU only (no GPU in the rule path)",
        },
        "command": "OMP_NUM_THREADS=2 python3 eval/exp_phys/run_determinism.py",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=1, sort_keys=False)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("hash_matrix", "receipt_sha256",
                                   "sample_receipt")}, indent=1))
    print("wrote", OUT)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
