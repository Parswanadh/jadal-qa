"""Q3 determinism proof: run the deterministic rule scorer 10x on a fixed set.

The 10 runs are **fresh processes** (one interpreter per run), not one long-lived
process. Per window, the receipt SHA-256 and the canonical receipt *bytes* must be
identical across all 10 fresh runs. Additional checks:
  * canonical (sorted-key) JSON is insensitive to feature dict insertion order;
  * no time/random/device keys and no floats anywhere in a receipt;
  * no GPU/CUDA/torch in the scorer's process (CPU only);
  * a **negative control** that perturbs one feature by one quantum (1e-4) and
    one threshold, and asserts the receipt hash and rule-table hash change;
  * the frozen input has **no missing features** (missing features fail closed).

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


def fresh_process_run(hashseed):
    """Run the whole scorer in a fresh interpreter; return per-window results.

    Returns ``{window_id: {"sha256": ..., "body": <canonical bytes hex>}}`` so the
    caller can compare full receipt bytes, not only the hash.
    """
    script = r"""
import hashlib, json, os, sys
sys.path.insert(0, %r); sys.path.insert(0, %r)
from decimal import Decimal
from eval.exp_phys import det_rule_scorer as D
with open(%r) as fh: inp = json.load(fh, parse_float=Decimal)
with open(%r) as fh: tab = json.load(fh)
ch = D.code_sha256()
out = {}
for w in inp["windows"]:
    r = D.score_window(w, tab, code_hash=ch)
    body = {k: v for k, v in r.items() if k != "receipt_sha256"}
    out[w["window_id"]] = {"sha256": r["receipt_sha256"],
                           "body": D.canonical_json(body).hex()}
print(json.dumps(out, sort_keys=True))
""" % (REPO, HERE, INPUT, RULES)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    env["OMP_NUM_THREADS"] = "2"
    p = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, env=env, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"fresh subprocess failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def main() -> None:
    data = load_input()
    windows = data["windows"]
    table = DRS.load_rule_table(RULES)
    code_hash = DRS.code_sha256()
    rtab_hash = DRS.rule_table_sha256(table)

    ids = [w["window_id"] for w in windows]

    # ---- 10 fresh-process runs (one interpreter per run) --------------------
    # PYTHONHASHSEED cycles 0/1/42 so hash randomisation is varied per run too.
    seeds = [HASHSEEDS[i % len(HASHSEEDS)] for i in range(N_RUNS)]
    fresh = [fresh_process_run(seed) for seed in seeds]

    hash_matrix = {wid: [fresh[r][wid]["sha256"] for r in range(N_RUNS)]
                   for wid in ids}
    body_matrix = {wid: [fresh[r][wid]["body"] for r in range(N_RUNS)]
                   for wid in ids}

    unique_per_window = {wid: len(set(hash_matrix[wid])) for wid in ids}
    n_identical = sum(1 for wid in ids if unique_per_window[wid] == 1)
    all_identical = n_identical == len(ids)

    # aggregate digest: sha256 of the concatenation (no separator) of the
    # per-window receipt hashes in sorted window_id order.
    aggregate_digest = hashlib.sha256(
        "".join(hash_matrix[wid][0] for wid in sorted(ids)).encode()
    ).hexdigest()

    # byte-identical bodies across runs (stronger than hash equality)
    bodies_identical = all(len(set(body_matrix[wid])) == 1 for wid in ids)

    # ---- one in-process pass, for schema invariants + sample only -----------
    inproc = run_once(windows, table, code_hash)
    sample_receipt = inproc[ids[0]][0]
    forbidden_ok = True
    for wid in ids:
        try:
            DRS.assert_no_forbidden_fields(inproc[wid][0])
        except ValueError:
            forbidden_ok = False
            break

    # ---- fail-closed: the frozen input must have no missing features --------
    missing_windows = {wid: inproc[wid][0]["missing_features"]
                       for wid in ids if inproc[wid][0]["missing_features"]}
    no_missing = not missing_windows

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

    # ---- cross-process: the 10 fresh runs themselves (seeds 0/1/42) ---------
    subprocess_ok = all_identical

    # ---- negative controls: a real change MUST change the receipt -----------
    # (a) perturb one feature by one quantum (1e-4) -> receipt hash must change
    w0 = windows[0]
    feat_name = table["rules"][0]["feature"]
    base_hash = inproc[w0["window_id"]][0]["receipt_sha256"]
    pert = {"window_id": w0["window_id"], "features": dict(w0["features"])}
    pert["features"][feat_name] = float(pert["features"][feat_name]) + 1e-4
    pert_hash = DRS.score_window(pert, table, code_hash=code_hash)["receipt_sha256"]
    feature_control_ok = (pert_hash != base_hash)

    # (b) perturb one threshold -> rule_table_sha256 (and the receipt) must change
    table2 = json.loads(json.dumps(table))
    table2["rules"][0]["thr_int"] = int(table2["rules"][0]["thr_int"]) + 1
    threshold_table_ok = (DRS.rule_table_sha256(table2) != rtab_hash)
    h2 = DRS.score_window(w0, table2, code_hash=code_hash)["receipt_sha256"]
    threshold_receipt_ok = (h2 != base_hash)
    threshold_control_ok = threshold_table_ok and threshold_receipt_ok

    # ---- decision distribution ----------------------------------------------
    dist = Counter(inproc[wid][0]["decision"].split(":")[0] for wid in ids)
    dist_by_split = {}
    for w in windows:
        d = inproc[w["window_id"]][0]["decision"].split(":")[0]
        dist_by_split.setdefault(w["split"], Counter())[d] += 1

    status_ok = (all_identical and bodies_identical and forbidden_ok
                 and order_ok and not gpu_modules and subprocess_ok
                 and feature_control_ok and threshold_control_ok and no_missing)

    result = {
        "experiment": "Q3.01",
        "title": "Byte-identical receipt proof for the deterministic rule path",
        "claim": ("On a fixed input set of real windows (150 gold IndEgo + 30 OOD "
                  "Egocentric-10K), the deterministic rule scorer produces the "
                  "same receipt SHA-256 for every window across 10 fresh-process "
                  "runs (same machine), and a deliberately perturbed input or "
                  "threshold does change the receipt."),
        "status": "PASS" if status_ok else "FAIL",
        "n_windows": len(ids),
        "n_gold": data["n_gold"],
        "n_ood": data["n_ood"],
        "n_runs": N_RUNS,
        "runs_are_fresh_processes": True,
        "n_fresh_processes": N_RUNS,
        "pythonhashseeds": seeds,
        "all_identical": bool(all_identical),
        "n_windows_identical": int(n_identical),
        "bodies_byte_identical": bool(bodies_identical),
        "aggregate_digest": aggregate_digest,
        "aggregate_digest_method": ("sha256(concat(sorted per-window "
                                     "receipt_sha256, no separator))"),
        "receipt_sha256": {wid: hash_matrix[wid][0] for wid in ids},
        "hash_matrix": hash_matrix,
        "unique_hashes_per_window": unique_per_window,
        "receipt_schema": {
            "body_fields": ["window_id", "decision", "evidence", "missing_features",
                            "feature_digest", "rule_table_sha256", "code_sha256", "q"],
            "envelope_fields": ["receipt_sha256"],
            "evidence_entry": ["rule", "feature", "value", "op", "threshold",
                               "missing"],
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
            "negative_control_feature_perturbation_changes_receipt": bool(feature_control_ok),
            "negative_control_threshold_changes_rule_table": bool(threshold_table_ok),
            "negative_control_threshold_changes_receipt": bool(threshold_receipt_ok),
            "no_missing_features_in_input": bool(no_missing),
        },
        "negative_control": {
            "feature_perturbed": feat_name,
            "feature_delta": 1e-4,
            "base_receipt_sha256": base_hash,
            "perturbed_receipt_sha256": pert_hash,
            "receipt_changed": bool(feature_control_ok),
            "threshold_rule": table["rules"][0]["name"],
            "rule_table_sha256_changed": bool(threshold_table_ok),
            "receipt_changed_on_threshold": bool(threshold_receipt_ok),
        },
        "missing_features_in_input": missing_windows,
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
