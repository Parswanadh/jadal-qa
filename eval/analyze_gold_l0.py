"""Funnel round 3 (issue #56): descriptive L0-vs-human-gold analysis.

Reads eval/results/gold_l0_signals.json + eval/results/gold_labels_snapshot.json
and routes every gold window through the SHIPPED thresholds (evidence_reject
evaluated for description; shipped default is integrity_only) to report,
with NO certificate attached:

  * verdict x human-label confusion at placeholder thresholds,
  * per-rule fired x human bad-reason cross-tab,
  * how much of the bad set is L0-checkable (TOO_DARK / DUPLICATE /
    UNDECODABLE directly; HANDS_NOT_VISIBLE via requires_hands under the
    assembly profile; NO_ACTIVE_WORK is NOT L0-checkable),
  * signal medians for good vs each bad-reason group (separation, not a bound).

Writes eval/results/funnel_gold_descriptive.json. Every number here is
exploratory: the certified numbers live in funnel_gold_calibration.json.

Usage:
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python eval/analyze_gold_l0.py
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import funnel  # noqa: E402

# Human bad reasons the L0 tier can check directly, per the round-3 brief.
L0_CHECKABLE = {"TOO_DARK", "DUPLICATE", "UNDECODABLE"}


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 4) if values else None


def main() -> int:
    # Fix round 1 (PR #59 review, Minor 2a): ROOT-anchored paths, so this
    # runs from any working directory.
    results = ROOT / "eval" / "results"
    cache = json.loads((results / "gold_l0_signals.json").read_text())
    labels = json.loads((results / "gold_labels_snapshot.json").read_text())
    shipped = funnel.load_thresholds()
    latest = labels["latest"]
    signals_by_id = cache["signals"]

    evidence_thresh = copy.deepcopy(shipped)
    evidence_thresh["hard_reject_policy"] = "evidence_reject"

    rows = []
    for seg_id, entry in sorted(latest.items()):
        signals = signals_by_id[seg_id]
        out_default = funnel.route(signals, None, evidence_thresh,
                                   allow_uncalibrated_accept=True)
        out_assembly = funnel.route(signals, None, evidence_thresh,
                                    allow_uncalibrated_accept=True,
                                    profile="assembly")
        rows.append({
            "segment_id": seg_id,
            "label": entry["verdict"],
            "reason_gold": entry.get("reason"),
            "verdict": out_default["verdict"],
            "reasons": out_default["reasons"],
            "verdict_assembly": out_assembly["verdict"],
            "reasons_assembly": out_assembly["reasons"],
        })

    confusion = Counter((r["label"], r["verdict"]) for r in rows)
    rule_hits: dict[str, Counter] = {}
    for r in rows:
        for reason in r["reasons"]:
            rule_hits.setdefault(reason, Counter())[r["reason_gold"] or
                                                    r["label"]] += 1

    bads = [r for r in rows if r["label"] == "bad"]
    checkable = [r for r in bads if r["reason_gold"] in L0_CHECKABLE]
    checkable_rejected = [r for r in checkable if r["verdict"] == "reject"]
    hands = [r for r in bads if r["reason_gold"] == "HANDS_NOT_VISIBLE"]
    hands_rejected_assembly = [r for r in hands
                               if r["verdict_assembly"] == "reject"]
    non_checkable = [r for r in bads
                     if r["reason_gold"] not in L0_CHECKABLE
                     and r["reason_gold"] != "HANDS_NOT_VISIBLE"]
    goods_rejected = [r for r in rows
                      if r["label"] == "good" and r["verdict"] == "reject"]

    numeric_fields = ["blur_var", "brightness_mean", "under_exposed_frac",
                      "over_exposed_frac", "motion_mean", "hands_visible_frac"]
    separation: dict[str, dict] = {}
    for field in numeric_fields:
        by_group: dict[str, list[float]] = {}
        for r in rows:
            value = signals_by_id[r["segment_id"]].get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            key = "good" if r["label"] == "good" else f"bad:{r['reason_gold']}"
            by_group.setdefault(key, []).append(float(value))
        separation[field] = {k: {"n": len(v), "median": _median(v)}
                             for k, v in sorted(by_group.items())}

    descriptive = {
        "snapshot_sha256": cache["labels_snapshot_sha256"],
        "n": len(rows),
        "n_good": sum(1 for r in rows if r["label"] == "good"),
        "n_bad": len(bads),
        "bad_reasons": dict(Counter(r["reason_gold"] for r in bads)),
        "thresholds": "shipped configs/funnel_thresholds.json, "
                      "evidence_reject evaluated for description "
                      "(shipped default integrity_only)",
        "confusion_label_x_verdict": {
            f"{label}/{verdict}": confusion[(label, verdict)]
            for label in ("good", "bad")
            for verdict in ("accept", "reject", "review")},
        "rule_fired_x_gold": {rule: dict(cnt)
                              for rule, cnt in sorted(rule_hits.items())},
        "l0_catch": {
            "checkable_bad_n": len(checkable),
            "checkable_rejected_n": len(checkable_rejected),
            "hands_bad_n": len(hands),
            "hands_rejected_under_assembly_n": len(hands_rejected_assembly),
            "non_l0_checkable_bad_n": len(non_checkable),
            "non_l0_checkable_reasons": dict(
                Counter(r["reason_gold"] for r in non_checkable)),
            "goods_falsely_rejected_n": len(goods_rejected),
        },
        "signal_medians_by_group": separation,
        "note": ("DESCRIPTIVE ONLY, no certificate: placeholder thresholds, "
                 "decisions None (no L1 yet), profile default unless noted."),
    }
    (results / "funnel_gold_descriptive.json").write_text(
        json.dumps(descriptive, indent=2, sort_keys=True) + "\n")
    print(json.dumps(descriptive, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
