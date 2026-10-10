"""LB1-rules verification: deterministic-rule + duplicate-detection claims (FACTS.md).

CPU-only, deterministic, reads only committed caches under eval/results/.
Writes eval/results/pod_verify/rules_and_duplicates.json.

Claims verified (exact wording in /workspace/jadal_pod_bundle/docs/FACTS.md):
  C1  funnel calibration / support floor 86 / CERTIFY 84 / TOO_DARK 6-of-6 at 1 FA
  C2  64-pair F1s (spectral 0.947 / EG2 0.909 / pHash 0.865); human-6 untested?
  C3  certified-error-bound arithmetic ~86 / 59 and effective sample size
  C4  degenerate deterministic rule list, rejected

Run: OMP_NUM_THREADS=2 python3 eval/verify/pod_rules_duplicates.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "eval", "results")
OUT_DIR = os.path.join(RES, "pod_verify")
sys.path.insert(0, ROOT)

from jadal import funnel  # noqa: E402
from jadal.spectral.dup import similarity  # noqa: E402

SIGNALS = os.path.join(RES, "gold_l0_signals.json")
GOLD_NPZ = os.path.join(RES, "perpixel", "gold.npz")
DUP_SIGS = os.path.join(RES, "perpixel", "dup_sigs.json")
M2 = os.path.join(RES, "perpixel", "m2.json")
RULELIST = os.path.join(RES, "perpixel", "rulelist.json")
CALIB = os.path.join(RES, "funnel_gold_calibration.json")
DESCR = os.path.join(RES, "funnel_gold_descriptive.json")
CONFIG = os.path.join(ROOT, "configs", "funnel_thresholds.json")


# ---------------------------------------------------------------- helpers
def ucb_zero(n: int, delta: float) -> float:
    """Exact one-sided Clopper-Pearson upper bound for 0 errors in n at
    confidence 1-delta (== funnel's floor formula, log(delta)/log(1-beta))."""
    return 1.0 - delta ** (1.0 / n)


def support_floor(delta_rule: float, beta: float) -> int:
    return math.ceil(math.log(delta_rule) / math.log(1.0 - beta))


def load_inputs() -> dict:
    with open(SIGNALS) as fh:
        sig = json.load(fh)
    z = np.load(GOLD_NPZ, allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    clips = [str(c) for c in z["clips"]]
    reasons = [str(r) for r in z["reasons"]]
    y = [int(v) for v in z["y"]]
    with open(os.path.join(RES, "gold_labels_snapshot.json")) as fh:
        labels = json.load(fh)["latest"]
    pod_manifest = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"
    cross = None
    if os.path.exists(pod_manifest):
        with open(pod_manifest) as fh:
            pm = json.load(fh)
        pmi = {it["id"]: it for it in pm["items"]}
        cross = {
            "path": pod_manifest,
            "labels_sha256": pm["labels_sha256"],
            "ids_match": sorted(pmi) == sorted(ids),
            "clips_match": all(str(pmi[i]["clip"]) == c
                               for i, c in zip(ids, clips)),
            "reasons_match": all((pmi[i].get("reason") or "") == r
                                 for i, r in zip(ids, reasons)),
            "labels_match": all(int(pmi[i]["label"]) == y
                                for i, y in zip(ids, y)),
        }
    return {"signals": sig["signals"], "ids": ids, "clips": clips,
            "reasons": reasons, "y": y, "labels": labels,
            "signals_path": SIGNALS, "gold_npz": GOLD_NPZ,
            "pod_manifest_cross_check": cross}


def build_records(inp: dict) -> list[dict]:
    """Same schema funnel.calibrate_two_sided consumes: signals + clip group,
    good/bad label, human_gold source, decisions None (no L1 confidences)."""
    out = []
    for wid, clip, y in zip(inp["ids"], inp["clips"], inp["y"]):
        sig = dict(inp["signals"][wid])
        out.append({
            "signals": sig,
            "decisions": None,
            "label": "good" if y == 1 else "bad",
            "reason_gold": inp["labels"][wid].get("reason"),
            "label_source": "human_gold",
            "group": clip,
            "segment_id": wid,
        })
    return out


# ---------------------------------------------------------------- C1 / C3
def claim_arithmetic() -> dict:
    beta = 0.05
    rows = {
        "shipped_bonferroni_delta0.1_m8": support_floor(0.1 / 8, beta),
        "shipped_fixed_sequence_delta0.1": support_floor(0.1, beta),
        "family_delta0.05_bonferroni_m4": support_floor(0.05 / 4, beta),
        "family_delta0.05_fixed_sequence": support_floor(0.05, beta),
    }
    ucb = {f"n{n}": round(ucb_zero(n, 0.0125), 5) for n in (84, 85, 86)}
    ucb59 = {f"n{n}": round(ucb_zero(n, 0.05), 5) for n in (58, 59)}
    return {"support_floors": rows, "ucb_delta_rule_0.0125": ucb,
            "ucb_delta_0.05": ucb59,
            "certify_split_84_passes": ucb_zero(84, 0.0125) <= beta,
            "certify_split_86_passes": ucb_zero(86, 0.0125) <= beta}


def claim_calibration(inp: dict) -> dict:
    records = build_records(inp)
    select, certify = funnel._split_select_certify(records, 1234)
    sel_groups = sorted({r["group"] for r in select})
    cert_groups = sorted({r["group"] for r in certify})
    result = funnel.calibrate_two_sided(records, {})
    mine = result["per_source"]["human_gold"]
    with open(CALIB) as fh:
        art = json.load(fh)
    a = art["per_source"]["human_gold"]
    rules_disabled = {k: (v["threshold"] is None)
                      for k, v in mine["reject_rules"].items()}
    art_disabled = {k: (v["threshold"] is None)
                    for k, v in a["reject_rules"].items()}
    return {
        "n_select": len(select), "n_certify": len(certify),
        "select_groups": sel_groups, "certify_groups": cert_groups,
        "meta": {"support_floor": result["meta"]["support_floor"],
                 "m_rules": result["meta"]["m_rules"],
                 "delta_per_rule": result["meta"]["delta_per_rule"]},
        "union_autonomous": {
            "errors": mine["union_autonomous"]["errors"],
            "fired": mine["union_autonomous"]["fired"],
            "upper_bound": mine["union_autonomous"]["upper_bound"]},
        "accept_candidate": mine["accept"]["candidate"],
        "rules_disabled": rules_disabled,
        "matches_artifact": {
            "n_select": len(select) == a["n_select"],
            "n_certify": len(certify) == a["n_certify"],
            "select_groups": sel_groups == a["select_groups"],
            "certify_groups": cert_groups == a["certify_groups"],
            "support_floor": result["meta"]["support_floor"]
            == art["meta"]["support_floor"],
            "rules_disabled": rules_disabled == art_disabled,
            "union_autonomous": (
                mine["union_autonomous"]["errors"]
                == a["union_autonomous"]["errors"]
                and mine["union_autonomous"]["upper_bound"]
                == a["union_autonomous"]["upper_bound"]),
            "accept_candidate_none": mine["accept"]["candidate"]
            == a["accept"]["candidate"] is None,
        },
        "artifact_values": {"n_select": a["n_select"], "n_certify": a["n_certify"],
                            "select_groups": a["select_groups"],
                            "certify_groups": a["certify_groups"],
                            "support_floor": art["meta"]["support_floor"]},
    }


def claim_too_dark(inp: dict) -> dict:
    with open(CONFIG) as fh:
        cfg = json.load(fh)
    thr_bright = float(cfg["brightness_min"])
    thr_under = float(cfg["under_exposed_frac_max"])
    fires = {}
    for wid, reason, y in zip(inp["ids"], inp["reasons"], inp["y"]):
        s = inp["signals"][wid]
        hit = (float(s["brightness_mean"]) < thr_bright
               or float(s["under_exposed_frac"]) > thr_under)
        if hit:
            key = reason or "GOOD"
            fires[key] = fires.get(key, 0) + 1
    n_dark = sum(1 for r in inp["reasons"] if r == "TOO_DARK")
    n_good = sum(1 for y in inp["y"] if y == 1)
    with open(DESCR) as fh:
        descr = {("GOOD" if k == "good" else k): v
                 for k, v in json.load(fh)["rule_fired_x_gold"]["TOO_DARK"].items()}
    return {
        "thresholds": {"brightness_min": thr_bright,
                       "under_exposed_frac_max": thr_under,
                       "code": "TOO_DARK (brightness_min lt OR under_exposed_frac_max gt)"},
        "fires_by_gold_reason": fires,
        "too_dark_recall": f"{fires.get('TOO_DARK', 0)}/{n_dark}",
        "good_false_alarms": fires.get("GOOD", 0),
        "n_good": n_good,
        "artifact_rule_fired_x_gold_TOO_DARK": descr,
        "matches_artifact": fires == descr,
    }


def claim_effective_n(inp: dict) -> dict:
    """No in-repo computation exists; this is an independent, explicitly
    labelled design-effect estimate from the 150 gold labels (cluster = clip),
    NOT the published protocol. ICC by one-way ANOVA on the binary outcome."""
    def design_effect(flag: np.ndarray) -> dict:
        clips = inp["clips"]
        uclips = sorted(set(clips))
        idx = {u: [i for i, c in enumerate(clips) if c == u] for u in uclips}
        groups = [flag[idx[u]] for u in uclips]
        n = len(flag)
        k = len(groups)
        ms = np.array([g.mean() for g in groups])
        ns = np.array([len(g) for g in groups], dtype=float)
        grand = flag.mean()
        ssb = float((ns * (ms - grand) ** 2).sum())
        ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
        msb, msw = ssb / (k - 1), ssw / (n - k)
        m0 = (n - (ns ** 2).sum() / n) / (k - 1)
        icc = (msb - msw) / (msb + (m0 - 1) * msw) if msb >= msw else 0.0
        mbar = n / k
        deff = 1.0 + (mbar - 1.0) * max(icc, 0.0)
        return {"icc": round(float(icc), 4), "design_effect": round(deff, 2),
                "n_eff": round(n / deff, 1)}
    bad = np.array([1.0 if y == 0 else 0.0 for y in inp["y"]])
    dark = np.array([1.0 if r == "TOO_DARK" else 0.0 for r in inp["reasons"]])
    return {"bad_rate": design_effect(bad), "too_dark_rate": design_effect(dark)}


# ---------------------------------------------------------------- C2
def claim_duplicates() -> dict:
    with open(DUP_SIGS) as fh:
        sigs = json.load(fh)
    with open(M2) as fh:
        m2 = json.load(fh)["duplicates"]
    z = np.load(GOLD_NPZ, allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    reasons = {str(i): str(r) for i, r in zip(z["ids"], z["reasons"])}
    clips = {str(i): str(c) for i, c in zip(z["ids"], z["clips"])}
    thr = float(m2["constructed"]["threshold"])
    dup_ids = sorted(i for i in ids if reasons[i] == "DUPLICATE")

    sims = {}
    nn = {}
    for a in ids:
        ranked = []
        for b in ids:
            if b == a:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in sims:
                sims[key] = similarity(sigs[a], sigs[b])
            ranked.append((b, sims[key]))
        ranked.sort(key=lambda t: -t[1])
        nn[a] = {"nn1": {"id": ranked[0][0], "sim": ranked[0][1],
                         "clip": clips[ranked[0][0]],
                         "reason": reasons[ranked[0][0]]},
                 "top3": [{"id": b, "sim": s, "clip": clips[b],
                           "reason": reasons[b]} for b, s in ranked[:3]]}
    rec1 = float(np.mean([nn[i]["nn1"]["sim"] >= thr for i in dup_ids]))
    rec3 = float(np.mean([any(t["sim"] >= thr for t in nn[i]["top3"])
                          for i in dup_ids]))
    flagged = [i for i in ids if nn[i]["nn1"]["sim"] >= thr]
    prec = len([i for i in flagged if reasons[i] == "DUPLICATE"]) / max(len(flagged), 1)

    f1 = {}
    for name, (p, r) in {"spectral": (1.0, 0.9), "eg2": (0.833, 1.0),
                         "phash": (0.941, 0.8)}.items():
        f1[name] = round(2 * p * r / (p + r), 4)
    missing = [p for p in ("/home/parshu/projects/jadal-qa-wt/spectral_data",
                           os.path.join(RES, "arms"))
               if not os.path.exists(p)]
    # bounded search for the constructed-pair inputs the three F1s need
    hits = []
    root = "/workspace"
    base_depth = root.count(os.sep)
    for base, dirs, files in os.walk(root):
        if base.count(os.sep) - base_depth >= 5:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in (".git", "arms")]
        for f in files:
            if f in ("manifest_dups.json", "dups.json", "spectral_study.md"):
                hits.append(os.path.join(base, f))
    pod_labels = "/workspace/jadal_pod_bundle/labels/manifest_gold.json"
    pairs_in_labels_manifest = False
    if os.path.exists(pod_labels):
        with open(pod_labels) as fh:
            pairs_in_labels_manifest = "pairs" in json.load(fh)
    return {
        "threshold_from_m2": thr,
        "dup_ids": dup_ids,
        "recomputed": {"recall_at1": rec1, "recall_at3": rec3,
                       "precision_top": prec, "n_flagged": len(flagged)},
        "m2_human": {"dup_ids": m2["human"]["dup_ids"],
                     "recall_at1": m2["human"]["recall_at1"],
                     "recall_at3": m2["human"]["recall_at3"],
                     "precision_top": m2["human"]["precision_top"],
                     "n_flagged": m2["human"]["n_flagged"]},
        "matches_artifact": (
            dup_ids == sorted(m2["human"]["dup_ids"])
            and abs(rec1 - m2["human"]["recall_at1"]) < 1e-12
            and abs(rec3 - m2["human"]["recall_at3"]) < 1e-12
            and abs(prec - m2["human"]["precision_top"]) < 1e-12
            and len(flagged) == m2["human"]["n_flagged"]),
        "nn_pairs": {i: nn[i]["nn1"] for i in dup_ids},
        "published_f1_internal_consistency": f1,
        "published_pairs": {"spectral": {"P": 1.0, "R": 0.9, "F1": 0.947},
                            "eg2": {"P": 0.833, "R": 1.0, "F1": 0.909},
                            "phash": {"P": 0.941, "R": 0.8, "F1": 0.865}},
        "missing_inputs_for_pair_level_recompute": missing,
        "constructed_pair_manifest_on_pod": {
            "pairs_key_in_pod_labels_manifest": pairs_in_labels_manifest,
            "bounded_search_maxdepth5": hits,
        },
    }


# ---------------------------------------------------------------- C4
def claim_rulelist(inp: dict) -> dict:
    with open(RULELIST) as fh:
        committed = json.load(fh)
    with open(CONFIG) as fh:
        cfg = json.load(fh)
    with open(CALIB) as fh:
        calib = json.load(fh)
    rerun = None
    try:
        import eval.perpixel.rules_fix as rf  # noqa: E402
        tmp = tempfile.mkdtemp(prefix="rules_fix_verify_")
        spec = os.path.join(tmp, "spec")
        res = os.path.join(tmp, "res")
        os.makedirs(spec)
        os.makedirs(res)
        items = [{"id": i, "clip": c, "reason": r or "",
                  "label": 1 if y == 1 else 0}
                 for i, c, r, y in zip(inp["ids"], inp["clips"],
                                       inp["reasons"], inp["y"])]
        with open(os.path.join(spec, "manifest_gold.json"), "w") as fh:
            json.dump({"items": items}, fh)
        for f in ("theory_feats.npz", "physics_feats.npz"):
            shutil.copy(os.path.join(RES, "perpixel", f), os.path.join(res, f))
        rf.SPEC_DIR = spec
        rf.RES = res
        rf.main()
        with open(os.path.join(res, "rulelist.json")) as fh:
            rerun = json.load(fh)
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as exc:  # recorded, never silent
        rerun = {"error": f"{type(exc).__name__}: {exc}"}
    fields = ("sens", "frr_good", "bal_acc", "per_reason", "beats_trivial")
    same = None
    if rerun and "error" not in rerun:
        same = all(rerun["loco"][k] == committed["loco"][k] for k in fields)
        same = same and rerun["active_rules"] == committed["active_rules"]
        same = same and rerun["fold_thresholds"] == committed["fold_thresholds"]
    rejects = {k: (v["threshold"] is None)
               for k, v in calib["per_source"]["human_gold"]["reject_rules"].items()}
    return {
        "committed_rulelist_loco": {k: committed["loco"][k] for k in fields},
        "committed_active_rules": committed["active_rules"],
        "rerun_loco": (None if rerun is None or "error" in rerun
                       else {k: rerun["loco"][k] for k in fields}),
        "rerun_error": (rerun or {}).get("error"),
        "rerun_identical_to_committed": same,
        "shipped_config": {"version": cfg["version"],
                           "calibration_status": cfg["calibration"]["status"],
                           "hard_reject_policy": cfg["hard_reject_policy"],
                           "accept_conf": cfg["accept_conf"]},
        "shipped_calibrated_rule_list_all_disabled": all(rejects.values()),
        "descriptive_evidence_reject_confusion": None,
    }


def main() -> int:
    inp = load_inputs()
    with open(DESCR) as fh:
        descr = json.load(fh)
    arith = claim_arithmetic()
    cal = claim_calibration(inp)
    dark = claim_too_dark(inp)
    effn = claim_effective_n(inp)
    dups = claim_duplicates()
    rules = claim_rulelist(inp)
    rules["descriptive_evidence_reject_confusion"] = \
        descr["confusion_label_x_verdict"]

    claims = [
        {
            "id": "LB1-rules-C1a",
            "text": ("Funnel calibration on the real signals: with 150 labels NO L0 "
                     "rule can be statistically certified (the multiple-testing "
                     "correction needs ~86 clean fires; the certification split has 84)."),
            "verdict": ("REPRODUCED"
                        if cal["matches_artifact"]["support_floor"]
                        and cal["matches_artifact"]["n_certify"]
                        and not arith["certify_split_84_passes"]
                        else "NOT REPRODUCED"),
            "published": {"required_clean_fires": 86, "certify_split": 84},
            "recomputed": {"support_floor": arith["support_floors"]
                           ["shipped_bonferroni_delta0.1_m8"],
                           "n_certify": cal["n_certify"],
                           "ucb_zero_of_84_at_delta_0.0125":
                               arith["ucb_delta_rule_0.0125"]["n84"],
                           "ucb_zero_of_86_at_delta_0.0125":
                               arith["ucb_delta_rule_0.0125"]["n86"],
                           "m_rules": cal["meta"]["m_rules"],
                           "delta_per_rule": cal["meta"]["delta_per_rule"],
                           "all_reject_rules_disabled":
                               all(cal["rules_disabled"].values()),
                           "accept_candidate": cal["accept_candidate"],
                           "matches_artifact": cal["matches_artifact"]},
            "artifact": "eval/results/funnel_gold_calibration.json",
            "command": ("OMP_NUM_THREADS=2 python3 eval/verify/"
                        "pod_rules_duplicates.py"),
            "protocol": ("floor = ceil(ln(delta/m)/ln(1-beta)) in jadal/funnel.py:"
                         "1216-1217 with delta=0.1, m=len(REJECT_RULES)=8, "
                         "beta=0.05; exact one-sided CP UCB(0/n)=1-delta^(1/n); "
                         "split by funnel._split_select_certify(records, 1234)"),
            "notes": ("86 = ceil(ln(0.0125)/ln(0.95)); UCB(0/84)@0.0125=0.0508>0.05 "
                      "so the 3-clip/84-window CERTIFY half cannot certify; "
                      "UCB(0/86)=0.0497<=0.05. Independent rerun reproduces the "
                      "artifact's n_select/n_certify/groups/support_floor and "
                      "all-8-rules-disabled."),
        },
        {
            "id": "LB1-rules-C1b",
            "text": ("Descriptively TOO_DARK catches 6/6 at 1 false alarm."),
            "verdict": "REPRODUCED" if dark["matches_artifact"] else "NOT REPRODUCED",
            "published": {"too_dark_recall": "6/6", "false_alarms": 1},
            "recomputed": {"too_dark_recall": dark["too_dark_recall"],
                           "good_false_alarms": dark["good_false_alarms"],
                           "fires_by_gold_reason": dark["fires_by_gold_reason"],
                           "matches_artifact": dark["matches_artifact"]},
            "artifact": "eval/results/funnel_gold_descriptive.json",
            "command": ("OMP_NUM_THREADS=2 python3 eval/verify/"
                        "pod_rules_duplicates.py"),
            "protocol": ("apply shipped placeholder TOO_DARK rules "
                         "(brightness_mean<40.0 or under_exposed_frac>0.35) to "
                         "the 150 cached SIGNALS; good/bad from gold_labels_snapshot"),
            "notes": ("6/6 TOO_DARK windows fire; 1 of 100 good windows fires "
                      "(plus 1 HANDS_NOT_VISIBLE, 1 NO_ACTIVE_WORK, i.e. the "
                      "false alarm count is over good windows). Matches the "
                      "artifact's rule_fired_x_gold row exactly."),
        },
        {
            "id": "LB1-rules-C2a",
            "text": ("Duplicates (64 constructed pairs, held-out half): spectral "
                     "signature F1 0.947, EmbeddingGemma-2 cosine 0.909, pHash 0.865 "
                     "(small n: differences not established)."),
            "verdict": "CANNOT VERIFY",
            "published": dups["published_pairs"],
            "recomputed": {"f1_from_published_PR":
                           dups["published_f1_internal_consistency"],
                           "pair_level_scores": None,
                           "constructed_pair_manifest_on_pod":
                               dups["constructed_pair_manifest_on_pod"]},
            "artifact": "none on this pod (source: legacy_models.md L1.08 -> "
                        "/home/parshu/projects/jadal-qa-wt/spectral-harness/docs/"
                        "reports/spectral_study.md, absent)",
            "command": "n/a (no pair manifest / per-detector scores present)",
            "protocol": ("would need the constructed-pair manifest and the three "
                         "detectors' held-out scores; the repo's own 128-pair "
                         "spectral rerun (L1.09) reports P 0.952/R 1.0/F1 0.976, "
                         "which the snapshot itself marks as not directly comparable"),
            "notes": ("F1s are internally consistent with the published P/R "
                      "(2PR/(P+R): 0.947/0.909/0.865 all follow) but the "
                      "underlying pairs are NOT on this pod: manifest_dups.json, "
                      "spectral_data/arms/B|E|A/*.json and spectral_study.md are "
                      "all absent; no script here reproduces these three numbers."),
        },
        {
            "id": "LB1-rules-C2b",
            "text": ("Not yet tested on the 6 real human DUPLICATE labels (in progress)."),
            "verdict": "NOT REPRODUCED",
            "published": {"status": "untested / in progress"},
            "recomputed": {**dups["recomputed"],
                           "dup_ids": dups["dup_ids"],
                           "matches_artifact": dups["matches_artifact"],
                           "nn_pairs": dups["nn_pairs"]},
            "artifact": ("eval/results/perpixel/m2.json (human section); "
                         "docs/reports/perpixel_accuracy.md; "
                         "eval/results/perpixel/dup_sigs.json (signatures)"),
            "command": ("OMP_NUM_THREADS=2 python3 eval/verify/"
                        "pod_rules_duplicates.py"),
            "protocol": ("recompute all 150x149 pairwise similarities from the "
                         "cached signatures with jadal.spectral.dup.similarity, "
                         "threshold 0.8410791366906474 taken from m2.json's "
                         "constructed-pair split; NN1/top3 per window; the 6 "
                         "DUPLICATE windows are g072/g077/g088/g089/g090/g149"),
            "notes": ("The claim is STALE: the 6 human DUPLICATE labels have been "
                      "tested. Independent recompute matches m2.json exactly: "
                      "recall@1=recall@3=0.5, precision of top matches=0.0476, "
                      "63/150 windows above the constructed threshold "
                      "(over-flagging). NN pairs for the 6 are listed in the JSON."),
        },
        {
            "id": "LB1-rules-C3a",
            "text": ("Certified error bounds: not achievable at n=150: the "
                     "multiple-testing correction needs ~86 clean windows (59 with "
                     "fixed-sequence testing alone)."),
            "verdict": "REPRODUCED",
            "published": {"bonferroni_clean_windows": 86,
                          "fixed_sequence_clean_windows": 59},
            "recomputed": arith["support_floors"],
            "artifact": ("jadal/funnel.py:1216-1217; "
                         "tests/test_funnel_round2.py:308 (86), :318 (45); "
                         "docs/reports/funnel.md:460; docs/pod/"
                         "SYNC_UPDATE_2026-10-10_2145.md:33 (K=4, 59)"),
            "command": "python3 -c \"import math; \
f=lambda d,b: math.ceil(math.log(d)/math.log(1-b)); print(f(0.0125,0.05), f(0.05,0.05))\"",
            "protocol": ("floor = ceil(ln(delta_rule)/ln(1-beta)), beta=0.05; "
                         "86 needs delta_rule=0.0125 (=0.1/8 or 0.05/4); "
                         "59 needs fixed-sequence delta=0.05"),
            "notes": ("Both figures follow from the same formula: 86 at "
                      "delta_rule=0.0125, 59 at family delta=0.05 with "
                      "fixed-sequence testing. CAVEAT: the shipped defaults use "
                      "delta=0.1, whose fixed-sequence opt-out floor is 45 "
                      "(asserted in tests/test_funnel_round2.py:318), not 59; "
                      "59 matches the out-of-repo research4 note (K=4, "
                      "docs/pod/SYNC_UPDATE...md:33). No artifact in this repo "
                      "computes 59 (docs/reports/claims_poc.md I1 records it as "
                      "unbacked and deleted from README)."),
        },
        {
            "id": "LB1-rules-C3b",
            "text": ("the independence assumption fails because windows within a "
                     "clip are correlated (effective sample size ~20-43)."),
            "verdict": "CANNOT VERIFY",
            "published": {"effective_sample_size_range": "20-43"},
            "recomputed": {"in_repo_computation": None,
                           "independent_design_effect_estimate": effn},
            "artifact": ("none: grep for effective sample / n_eff / design effect / "
                         "ICC finds no computation; provenance is the out-of-repo "
                         "'research4 risk-control note' "
                         "(docs/pod/SYNC_UPDATE_2026-10-10_2145.md:33)"),
            "command": "grep -rn 'effective sample|n_eff|design effect|ICC' ...",
            "protocol": ("published protocol not stated on the pod; my independent "
                         "check is NOT it: one-way-ANOVA ICC over 7 clip clusters "
                         "on the binary outcome, n_eff=n/(1+(mbar-1)ICC)"),
            "notes": ("No computation exists in-repo (claims_poc.md I1 lists "
                      "'20-43' among fragments with no backing). The pod's own "
                      "sync note gives a DIFFERENT range, ~11-43, from the "
                      "research4 note. My labelled cross-check lands at "
                      "n_eff(bad rate) and n_eff(TOO_DARK rate) reported in the "
                      "JSON; it is a plausibility check only."),
        },
        {
            "id": "LB1-rules-C4",
            "text": ("The deterministic threshold rule list tried so far is "
                     "degenerate (rejects nearly everything) and was rejected."),
            "verdict": ("REPRODUCED" if rules["rerun_identical_to_committed"]
                        else "NOT REPRODUCED"),
            "published": {"behaviour": "degenerate, rejects nearly everything",
                          "status": "rejected"},
            "recomputed": {"loco": rules["rerun_loco"],
                           "active_rules": rules["committed_active_rules"],
                           "rerun_identical": rules["rerun_identical_to_committed"],
                           "rerun_error": rules["rerun_error"]},
            "artifact": ("eval/results/perpixel/rulelist.json; "
                         "eval/results/perpixel/theory.json rule_list; "
                         "eval/perpixel/rules_fix.py docstring; "
                         "docs/reports/perpixel_accuracy.md"),
            "command": ("OMP_NUM_THREADS=2 python3 eval/verify/"
                        "pod_rules_duplicates.py (reruns rules_fix.main() into a "
                        "temp dir with manifest_gold rebuilt from gold.npz)"),
            "protocol": ("deterministic per-rule thresholds, FRR cap 0.15, nested "
                         "LOCO over the 7 clips on the quality task "
                         "(TOO_DARK+UNDECODABLE vs good)"),
            "notes": ("Evidence is recorded and the rerun is bit-identical: "
                      "sens 1.0, frr_good 0.97 (97/100 good windows rejected), "
                      "bal_acc 0.515, GOOD kept 0.03. The pre-fix objective "
                      "(unconstrained Youden) kept 0.0 good (rules_fix.py "
                      "docstring; theory.json rule_list bal_acc 0.47, AUROC "
                      "0.589). STILL TRUE OF THE SHIPPED LIST? The shipped "
                      "configs/funnel_thresholds.json is version "
                      "'funnel-v1-placeholder', status placeholder, "
                      "hard_reject_policy integrity_only, and the round-2 "
                      "calibration DISABLES all 8 tunable reject rules; under "
                      "evidence_reject the shipped thresholds descriptively "
                      "reject 12/100 good windows (not 'nearly everything'). So "
                      "the degenerate list is correctly recorded as tried and "
                      "rejected, and it is not what ships."),
        },
    ]
    payload = {
        "meta": {
            "task": "LB1-rules verification (deterministic rules + duplicates)",
            "claims_source": "/workspace/jadal_pod_bundle/docs/FACTS.md "
                             "(lines 22, 23, 41, 46)",
            "repo_sha": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
            "cpu_only": True,
            "inputs_readonly": ["/workspace/jadal_pod_bundle/**",
                                "other /workspace/wt/* worktrees"],
            "inputs": [os.path.relpath(p, ROOT) for p in
                       (SIGNALS, GOLD_NPZ, DUP_SIGS, M2, RULELIST, CALIB, DESCR,
                        CONFIG)],
            "pod_manifest_cross_check": inp["pod_manifest_cross_check"],
            "generated_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
        },
        "claims": claims,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "rules_and_duplicates.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    for c in claims:
        print(f"{c['id']}: {c['verdict']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
