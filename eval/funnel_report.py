"""Funnel split + QA-enrichment report (GitHub issue #26, fix round 1).

Routes every segment with jadal.funnel.route and prints/saves: the funnel
split (% reject / accept / review), and — per label source, never pooled —
the exact-binomial FALSE-REJECT upper bound (good among rejected), the
DEFECT upper bound among accepted, the QA-queue ENRICHMENT (defect rate
among review+accept vs unfiltered) and the QA WORKLOAD reduction
(1 - review share). Without labels, bound sections are skipped (said
plainly). Counts (split) may be shown overall; every CERTIFICATE is
per-source.

Optional parametric QA-cost illustration: pass --minutes-per-review (USER
supplied) to convert the review queue into QA hours. No Awign numbers are
invented or assumed.

Usage:
    PYTHONNOUSERSITE=1 python3 eval/funnel_report.py \\
        --synthetic 1200 --seed 7 --allow-uncalibrated
    (records: JSON list of {signals, decisions, label, label_source}; or
     --synthetic N for deterministic synthetic fixtures, labelled synthetic)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import certify as certify_mod  # noqa: E402
from jadal import funnel  # noqa: E402
from eval.calibrate_funnel import synthetic_records  # noqa: E402

DELTA = 0.1


def _bound(errors: int, total: int) -> float | None:
    if not total:
        return None
    return round(certify_mod.exact_binomial_upper(errors, total, DELTA), 4)


def _source_section(routed: list[dict], source: str,
                    minutes_per_review: float | None) -> dict:
    rows = [r for r in routed if r.get("label_source") == source]
    total = len(rows)
    split = {v: sum(1 for r in rows if r["verdict"] == v)
             for v in ("accept", "reject", "review")}
    section: dict = {
        "n": total,
        "split": split,
        "split_pct": {v: round(split[v] / total * 100, 2) for v in split}
        if total else {},
        "review_share": round(split["review"] / total, 4) if total else None,
        "qa_workload_reduction": round(1 - split["review"] / total, 4)
        if total else None,
    }
    labelled = [r for r in rows if r.get("label") in ("good", "bad")]
    section["n_labelled"] = len(labelled)
    if labelled:
        rej = [r for r in labelled if r["verdict"] == "reject"]
        acc = [r for r in labelled if r["verdict"] == "accept"]
        queue = [r for r in labelled if r["verdict"] in ("review", "accept")]
        rej_good = sum(1 for r in rej if r["label"] == "good")
        acc_bad = sum(1 for r in acc if r["label"] == "bad")
        queue_bad = sum(1 for r in queue if r["label"] == "bad")
        all_bad = sum(1 for r in labelled if r["label"] == "bad")
        section["false_reject"] = {
            "rejected": len(rej), "good_among_rejected": rej_good,
            "upper_bound": _bound(rej_good, len(rej)),
        }
        section["defect_among_accepted"] = {
            "accepted": len(acc), "bad_among_accepted": acc_bad,
            "upper_bound": _bound(acc_bad, len(acc)),
        }
        unfiltered = all_bad / len(labelled)
        enriched = (queue_bad / len(queue)) if queue else None
        section["enrichment"] = {
            "defect_rate_unfiltered": round(unfiltered, 4),
            "defect_rate_queue": (round(enriched, 4)
                                  if enriched is not None else None),
            "lift": (round(enriched / unfiltered, 3)
                     if enriched and unfiltered else None),
        }
    else:
        section["note"] = "no labels: bounds and enrichment skipped"
    if minutes_per_review is not None and total:
        hours_all = total * minutes_per_review / 60
        hours_queue = split["review"] * minutes_per_review / 60
        section["qa_hours"] = {
            "minutes_per_review": minutes_per_review,
            "hours_watch_everything": round(hours_all, 2),
            "hours_review_queue": round(hours_queue, 2),
            "hours_saved": round(hours_all - hours_queue, 2),
        }
    return section


def report(records: list[dict], thresholds: dict,
           minutes_per_review: float | None = None,
           allow_uncalibrated: bool = False,
           allow_provisional: bool = False,
           profile: str = "default") -> dict:
    routed = []
    for record in records:
        out = funnel.route(record.get("signals"), record.get("decisions"),
                           thresholds,
                           allow_uncalibrated_accept=allow_uncalibrated,
                           allow_provisional_accept=allow_provisional,
                           profile=profile)
        routed.append({**record, **out})
    total = len(routed)
    split = {v: sum(1 for r in routed if r["verdict"] == v)
             for v in ("accept", "reject", "review")}
    sources: dict[str, int] = {}
    for record in routed:
        sources[record.get("label_source", "unlabeled")] = \
            sources.get(record.get("label_source", "unlabeled"), 0) + 1
    calibration = thresholds.get("calibration", {})
    return {
        "n": total,
        "label_sources": sources,
        "split": split,  # counts only, not a certificate
        "split_pct": {v: round(split[v] / total * 100, 2) for v in split},
        "profile": profile,
        "hard_reject_policy": thresholds.get("hard_reject_policy",
                                             "integrity_only"),
        "calibration_status": calibration.get("status"),
        "calibration_label_source": calibration.get("label_source"),
        # Certificates below are PER SOURCE, never pooled.
        "per_source": {s: _source_section(routed, s, minutes_per_review)
                       for s in sorted(sources)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--records", type=Path)
    group.add_argument("--synthetic", type=int, metavar="N")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--thresholds", type=Path,
                        default=ROOT / "configs" / "funnel_thresholds.json")
    parser.add_argument("--allow-uncalibrated", action="store_true",
                        help="pass allow_uncalibrated_accept=True to route() "
                             "(demo on placeholder thresholds)")
    parser.add_argument("--allow-provisional", action="store_true",
                        help="pass allow_provisional_accept=True to route() "
                             "(demo on provisional thresholds; every accept "
                             "carries accept_status evidence)")
    parser.add_argument("--profile", type=str, default="default",
                        help="task profile for the hands rule "
                             "(e.g. assembly, logistics)")
    parser.add_argument("--minutes-per-review", type=float, default=None,
                        help="USER-supplied QA minutes per review clip")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "eval" / "results" / "funnel_report.json")
    args = parser.parse_args(argv)

    if args.records:
        with open(args.records, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload if isinstance(payload, list) \
            else payload.get("records", [])
    else:
        records = synthetic_records(args.synthetic, args.seed)
    with open(args.thresholds, "r", encoding="utf-8") as fh:
        thresholds = json.load(fh)
    result = report(records, thresholds, args.minutes_per_review,
                    allow_uncalibrated=args.allow_uncalibrated,
                    allow_provisional=args.allow_provisional,
                    profile=args.profile)
    result["thresholds_version"] = thresholds.get("version")
    result["provenance"] = ("synthetic-fixtures" if args.records is None
                            else f"records-file:{args.records}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"n={result['n']} sources={result['label_sources']} "
          f"split={result['split']} ({result['split_pct']})")
    for source, section in result["per_source"].items():
        print(f"[{source}] n={section['n']} split={section['split']} "
              f"FR={section.get('false_reject')} "
              f"defect={section.get('defect_among_accepted')} "
              f"enrichment={section.get('enrichment')} "
              f"workload_reduction={section.get('qa_workload_reduction')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
