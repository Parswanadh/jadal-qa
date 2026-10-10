"""Two-sided funnel calibration driver (GitHub issue #26, Phase 2).

Reads labelled records ``{signals, decisions|None, label, reason_gold,
label_source}`` from a JSON file, or generates deterministic SYNTHETIC
fixtures (``--synthetic N``) to exercise the pipeline end to end, then runs
``jadal.funnel.calibrate_two_sided`` and writes
``eval/results/funnel_calibration.json``.

Label honesty: synthetic fixtures are labelled ``label_source:
'synthetic'`` and every bound is reported per source. A synthetic run NEVER
overwrites ``configs/funnel_thresholds.json`` unless ``--write-config`` is
passed explicitly (human gold runs should pass it).

Usage:
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        eval/calibrate_funnel.py --synthetic 400 --seed 7
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        eval/calibrate_funnel.py --records eval/gold/labels.json --write-config
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import funnel  # noqa: E402

RESULTS_PATH = ROOT / "eval" / "results" / "funnel_calibration.json"
CONFIG_PATH = ROOT / "configs" / "funnel_thresholds.json"


def synthetic_records(n: int, seed: int) -> list[dict]:
    """Deterministic synthetic records (label_source='synthetic').

    Good clips sit comfortably inside every L0 limit with confident heads;
    bad clips violate 1-3 random rules with weak heads. Margins are wide on
    purpose: this mode validates the calibration MACHINERY, not real
    thresholds — never present its numbers as human-grade.
    """
    rng = random.Random(seed)
    rules = ["blur", "dark", "over", "shaky", "static", "hands"]
    records: list[dict] = []
    for _ in range(n):
        bad = rng.random() < 0.4
        signals = {
            "decodable": True,
            "duration_s": 10.0,
            "blur_var": rng.uniform(90, 220),
            "brightness_mean": rng.uniform(80, 160),
            "under_exposed_frac": rng.uniform(0.0, 0.05),
            "over_exposed_frac": rng.uniform(0.0, 0.05),
            "motion_mean": rng.uniform(2.0, 8.0),
            "hands_visible_frac": rng.uniform(0.7, 1.0),
            "dup_of": None,
        }
        reason_gold = None
        if bad:
            for rule in rng.sample(rules, rng.randint(1, 3)):
                if rule == "blur":
                    signals["blur_var"] = rng.uniform(5, 40)
                    reason_gold = "BLUR"
                elif rule == "dark":
                    signals["brightness_mean"] = rng.uniform(5, 30)
                    reason_gold = reason_gold or "TOO_DARK"
                elif rule == "over":
                    signals["over_exposed_frac"] = rng.uniform(0.5, 0.9)
                    reason_gold = reason_gold or "OVEREXPOSED"
                elif rule == "shaky":
                    signals["motion_mean"] = rng.uniform(20, 60)
                    reason_gold = reason_gold or "SHAKY"
                elif rule == "static":
                    signals["motion_mean"] = rng.uniform(0.0, 0.2)
                    reason_gold = reason_gold or "NO_ACTIVE_WORK"
                elif rule == "hands":
                    signals["hands_visible_frac"] = rng.uniform(0.0, 0.3)
                    reason_gold = reason_gold or "HANDS_NOT_VISIBLE"
        lo, hi = (0.05, 0.6) if bad else (0.93, 0.999)
        records.append({
            "signals": signals,
            "decisions": {
                "hands_ok": {"p_yes": round(rng.uniform(lo, hi), 4)},
                "active_work": {"p_yes": round(rng.uniform(lo, hi), 4)},
                "step": {"choice": "assemble", "probs": {},
                         "answer_conf": round(rng.uniform(lo, hi), 4)},
            },
            "label": "bad" if bad else "good",
            "reason_gold": reason_gold,
            "label_source": "synthetic",
        })
    return records


def apply_to_config(result: dict, path: Path) -> dict:
    """Merge calibrated thresholds into the config file; recompute sha256."""
    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    updates = result["threshold_updates"]
    disabled = [k for k, v in updates.items()
                if v is None and k != "accept_conf"]
    for key, value in updates.items():
        config[key] = value
    meta = result["meta"]
    accept = result["accept"]
    config["calibration"] = {
        "status": ("calibrated" if meta["by_source"].get("human_gold")
                   else "synthetic-only"
                   if set(meta["by_source"]) == {"synthetic"}
                   else "mixed-nonhuman"),
        "label_sources": meta["by_source"],
        "n_total": meta["n_total"],
        "n_calib": meta["n_calib"],
        "n_heldout": meta["n_heldout"],
        "targets": meta["targets"],
        "seed": meta["seed"],
        "date_utc": meta["date_utc"],
        "accept_heldout": accept.get("heldout"),
        "accept_heldout_by_source": accept.get("heldout_by_source"),
        "disabled_rules": disabled,
        "note": ("Bounds carry only the certificate of their label source. "
                 "Synthetic/Gemini numbers are machinery checks, never "
                 "human-grade claims."),
    }
    config["sha256"] = funnel.thresholds_hash(config)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--records", type=Path,
                       help="JSON file with a list of labelled records")
    group.add_argument("--synthetic", type=int, metavar="N",
                       help="generate N deterministic synthetic records")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--min-accepted", type=int, default=30)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--write-config", action="store_true",
                        help="also merge the result into the thresholds file")
    args = parser.parse_args(argv)

    if args.records:
        with open(args.records, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload if isinstance(payload, list) \
            else payload.get("records", [])
    else:
        records = synthetic_records(args.synthetic, args.seed)
    result = funnel.calibrate_two_sided(records, {
        "target_alpha": args.alpha, "target_beta": args.beta,
        "delta": args.delta, "min_support": args.min_support,
        "min_accepted": args.min_accepted, "seed": args.seed})
    result["meta"]["provenance"] = (
        "synthetic-fixtures" if args.records is None
        else f"records-file:{args.records}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    accept = result["accept"]
    print(f"records={result['meta']['n_total']} "
          f"(calib={result['meta']['n_calib']}, "
          f"heldout={result['meta']['n_heldout']}) "
          f"sources={result['meta']['by_source']}")
    print(f"accept_conf={accept['threshold']} "
          f"heldout={accept['heldout']}")
    for key, entry in result["reject_rules"].items():
        print(f"  {key}: threshold={entry['threshold']} "
              f"disabled={entry['disabled']} "
              f"heldout={entry['heldout']}")
    print(f"wrote {args.out}")
    if args.write_config:
        apply_to_config(result, args.config)
        print(f"updated {args.config}")
    else:
        print("(config file untouched: pass --write-config to update it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
