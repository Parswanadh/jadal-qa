"""Jadal QA: calibration report for predictions on the ego10k eval split.

Usage (from the repo root):

    /home/parshu/envs/jadal/bin/python -m eval.calibration_report \
        --predictions eval/results/predictions.json \
        --splits eval/results/splits.json \
        --select-split calibration --report-split test

The accept threshold is chosen on the SELECTION set only (default: the
calibration split) with jadal.certify.pick_threshold (fixed-sequence testing,
exact-binomial certificate at --delta), and accuracy / ECE / the
risk-coverage curve are reported on the REPORT set (default: --report-split
test). Selection and report frame_ids must be disjoint — the tool refuses
with an error otherwise, so a threshold is never tuned on the set it is
reported on.

Input (a JSON file): a list of
``{"frame_id": str, "probs": float | [p_no, p_yes] | {"yes": p, "no": q} | {"0": .., "1": ..}, "label": "yes"|"no"|0|1}``.
``label`` is the (not human-checked) gemini-2.5-flash auto-label read from the
parquet files; see ``eval/README.md``. Records with a missing/unparseable
``probs`` or ``label`` are counted under ``skipped`` and excluded.

Outputs:
* ``eval/results/calibration.json``  - accuracy-vs-coverage points, ECE, and the
  threshold that meets a target error (if any).
* ``eval/results/calibration.png``   - reliability diagram + accuracy/error vs
  coverage curve (matplotlib, ``Agg`` backend, no display needed).

``jadal.certify`` (``jadal/certify.py``: ``ece``, ``risk_coverage``,
``pick_threshold``) is used when it is importable; otherwise this module falls
back to local implementations of exactly those formulas. The report records
which one produced each number.

NOT DONE:
- No baseline model ships here yet, so this run reports on whatever predictions
  file the caller supplies (a schema example is written by ``--write-example``).
- Conformal / exact-binomial error bounds are NOT computed here; they belong to
  ``jadal.certify.exact_binomial_upper`` which does not exist yet on the branch
  this module was written against.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
ROOT = EVAL_DIR.parent

FLOAT_DECIMALS = 4  # SPEC: floats are rounded to 4 decimals before entering a report/receipt
DEFAULT_BINS = 15
DEFAULT_TARGET_ERROR = 0.05
DEFAULT_MIN_ACCEPTED = 30

POSITIVE_LABELS = {"yes", "y", "true", "1", 1, True, 1.0}
NEGATIVE_LABELS = {"no", "n", "false", "0", 0, False, 0.0}


def log(msg: str) -> None:
    print(f"[calibration_report] {msg}", flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _round(x: float) -> float:
    return round(float(x) + 0.0, FLOAT_DECIMALS)


# ---------------------------------------------------------------------------
# certify import (optional)
# ---------------------------------------------------------------------------


def _load_certify():
    """Return jadal.certify if it exists on this branch, else None."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from jadal import certify  # type: ignore

        return certify
    except Exception as exc:  # ModuleNotFoundError, ImportError, anything else
        log(f"jadal.certify not importable ({type(exc).__name__}: {exc}); using local fallback")
        return None


# ---------------------------------------------------------------------------
# prediction parsing
# ---------------------------------------------------------------------------


def _as_p_yes(probs: Any) -> float | None:
    """Normalise a probability field into p(label == yes)."""
    if probs is None:
        return None
    if isinstance(probs, bool):
        return None
    if isinstance(probs, (int, float)):
        p = float(probs)
    elif isinstance(probs, dict):
        keys = {str(k).lower(): v for k, v in probs.items()}
        for key in ("yes", "1", "true", "p_yes", "positive"):
            if key in keys:
                try:
                    return _clip(float(keys[key]))
                except (TypeError, ValueError):
                    return None
        if len(keys) == 2:
            # Two-key dict without a recognised positive key: order is
            # arbitrary, so refuse instead of guessing which entry is P(yes).
            return None
        return None
    elif isinstance(probs, (list, tuple)):
        if len(probs) == 2:
            try:
                return _clip(float(probs[1]))
            except (TypeError, ValueError):
                return None
        if len(probs) == 1:
            return _clip(float(probs[0]))
        return None
    else:
        return None
    if math.isnan(p) or math.isinf(p):
        return None
    return _clip(p)


def _clip(p: float) -> float:
    return min(1.0, max(0.0, p))


def _as_label(label: Any) -> int | None:
    if label is None:
        return None
    if isinstance(label, str):
        key = label.strip().lower()
        if key in POSITIVE_LABELS:
            return 1
        if key in NEGATIVE_LABELS:
            return 0
        return None
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)) and not isinstance(label, bool):
        if label == 1:
            return 1
        if label == 0:
            return 0
        if 0.0 < float(label) < 1.0:
            # A soft label in (0, 1) is binarised at 0.5 (documented choice;
            # callers needing strict labels should pre-threshold upstream).
            return 1 if float(label) >= 0.5 else 0
        return None
    return None


def load_predictions(path: Path) -> tuple[list[dict], int]:
    """Read a predictions file, returning (records, skipped_count)."""
    data = json.loads(path.read_text())
    raw: Sequence[Any]
    if isinstance(data, dict):
        if isinstance(data.get("predictions"), list):
            raw = data["predictions"]
        else:  # {"frame_id": {...}} mapping
            raw = [{"frame_id": fid, **(v if isinstance(v, dict) else {"probs": v})} for fid, v in data.items()]
    elif isinstance(data, list):
        raw = data
    else:
        raise SystemExit(f"unexpected predictions structure in {path}")
    records: list[dict] = []
    skipped = 0
    for item in raw:
        if not isinstance(item, dict):
            skipped += 1
            continue
        fid = item.get("frame_id", item.get("id"))
        raw_probs = item.get("probs", item.get("prob", item.get("p_yes", item.get("score"))))
        if isinstance(raw_probs, dict) and "p_yes" in raw_probs:
            p_yes = _as_p_yes(raw_probs["p_yes"])
        else:
            p_yes = _as_p_yes(raw_probs)
        label = item.get("label", item.get("gold", item.get("y_true", item.get("active_labor"))))
        y = _as_label(label)
        if fid is None or p_yes is None or y is None:
            skipped += 1
            continue
        records.append({"frame_id": str(fid), "p_yes": float(p_yes), "y": int(y)})
    return records, skipped


def restrict_to_split(records: list[dict], splits_path: Path, split_name: str) -> tuple[list[dict], int]:
    manifest = json.loads(splits_path.read_text())
    splits = manifest.get("splits", manifest)
    if split_name not in splits:
        raise SystemExit(f"split {split_name!r} not found in {splits_path} (have {sorted(splits)})")
    allowed = set(splits[split_name]["frame_ids"])
    kept = [r for r in records if r["frame_id"] in allowed]
    dropped = len(records) - len(kept)
    if dropped:
        log(f"restricted to split {split_name!r}: kept {len(kept)}, dropped {dropped} frame_ids outside the split")
    return kept, dropped


# ---------------------------------------------------------------------------
# local fallbacks (mirror the jadal/certify.py contract in docs/SPEC.md)
# ---------------------------------------------------------------------------


def pred_conf(p_yes: float) -> tuple[int, float]:
    """Argmax prediction and its confidence for a binary p_yes."""
    pred = 1 if float(p_yes) >= 0.5 else 0
    return pred, max(float(p_yes), 1.0 - float(p_yes))


def bin_stats(conf: Sequence[float], correct: Sequence[int], bins: int) -> list[dict]:
    counts = [0] * bins
    conf_sum = [0.0] * bins
    corr = [0] * bins
    for c, ok in zip(conf, correct):
        idx = min(bins - 1, max(0, int(_clip(float(c)) * bins)))
        counts[idx] += 1
        conf_sum[idx] += float(c)
        corr[idx] += int(ok)
    out = []
    for idx in range(bins):
        if not counts[idx]:
            continue
        out.append(
            {
                "bin": idx,
                "n": counts[idx],
                "conf": _round(conf_sum[idx] / counts[idx]),
                "acc": _round(corr[idx] / counts[idx]),
            }
        )
    return out


def local_ece(probs: Sequence[float], y: Sequence[int], bins: int = DEFAULT_BINS) -> float:
    """ECE with equal-width bins over the *predicted-class* confidence.

    For binary probs p = P(y=1): confidence c = max(p, 1-p), correctness =
    (1 if p >= 0.5 else 0) == y. ECE = sum_b (n_b/N) |acc_b - conf_b|.
    """
    if not probs:
        return 0.0
    stats = bin_stats(
        [max(float(p), 1.0 - float(p)) for p in probs],
        [int((1 if float(p) >= 0.5 else 0) == int(yi)) for p, yi in zip(probs, y)],
        bins,
    )
    n = len(probs)
    return _round(sum(s["n"] / n * abs(s["acc"] - s["conf"]) for s in stats))


def local_risk_coverage(conf: Sequence[float], correct: Sequence[int]) -> list[dict]:
    """Accuracy-vs-coverage points for "accept if conf >= threshold"."""
    n = len(conf)
    if n == 0:
        return []
    pairs = sorted(zip((float(c) for c in conf), (int(c) for c in correct)), key=lambda t: (-t[0], t[1]))
    thresholds = sorted({p for p, _ in pairs}, reverse=True)
    out: list[dict] = []
    for tau in thresholds:
        accepted = [(p, c) for p, c in pairs if p >= tau]
        if not accepted:
            continue
        cov = len(accepted) / n
        err = 1.0 - sum(c for _, c in accepted) / len(accepted)
        out.append({"threshold": _round(tau), "coverage": _round(cov), "error": _round(err)})
    # the "accept everything" point
    out.append({"threshold": _round(0.0), "coverage": _round(1.0), "error": _round(1.0 - sum(c for _, c in pairs) / n)})
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _conf_correct(records: Sequence[dict]) -> tuple[list[float], list[int]]:
    """Predicted-class confidences and argmax-correctness flags."""
    conf = [r["p_yes"] for r in records]
    y = [r["y"] for r in records]
    pred = [int(p >= 0.5) for p in conf]
    correct = [int(pr == int(yi)) for pr, yi in zip(pred, y)]
    confidence = [max(float(p), 1.0 - float(p)) for p in conf]
    return confidence, correct


def local_binomial_upper(errors: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound (local mirror of certify)."""
    from scipy.stats import beta

    errors, n = int(errors), int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if errors == 0:
        return 1.0 - delta ** (1.0 / n)
    if errors == n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def local_pick_threshold(
    conf: Sequence[float],
    correct: Sequence[int],
    target_err: float,
    delta: float = 0.1,
    min_accepted: int = DEFAULT_MIN_ACCEPTED,
) -> dict:
    """Fixed-sequence threshold scan (local mirror of certify.pick_threshold).

    Candidates are the distinct confidences, strictest first; the scan stops
    at the first candidate that accepts fewer than min_accepted or whose
    exact-binomial error upper bound exceeds target_err, and returns the last
    passer. Same return schema as jadal.certify.pick_threshold.
    """
    c = [float(x) for x in conf]
    ok = [int(x) for x in correct]
    total = len(c)
    if total == 0:
        return {"threshold": None, "accepted": 0, "errors": 0, "upper_bound": None, "coverage": 0.0}
    best: dict | None = None
    for threshold in sorted(set(c), reverse=True):
        accepted = sum(1 for v in c if v >= threshold)
        errors = sum(1 for v, o in zip(c, ok) if v >= threshold and not o)
        upper = local_binomial_upper(errors, accepted, delta)
        if accepted < min_accepted or upper > target_err:
            break
        best = {
            "threshold": _round(threshold),
            "accepted": accepted,
            "errors": errors,
            "upper_bound": _round(upper),
            "coverage": _round(accepted / total),
        }
    if best is not None:
        return best
    errors = sum(1 for o in ok if not o)
    return {
        "threshold": None,
        "accepted": total,
        "errors": errors,
        "upper_bound": _round(local_binomial_upper(errors, total, delta)),
        "coverage": 1.0,
    }


def select_threshold(
    select_records: Sequence[dict],
    certify,
    target_error: float,
    delta: float,
    min_accepted: int,
) -> tuple[dict, str]:
    """Choose the accept threshold on the SELECTION set only."""
    confidence, correct = _conf_correct(select_records)
    if certify is not None and hasattr(certify, "pick_threshold"):
        try:
            res = certify.pick_threshold(
                confidence, correct,
                target_err=target_error, delta=delta, min_accepted=min_accepted,
            )
            out = {
                "threshold": res.get("threshold"),
                "accepted": res.get("accepted"),
                "errors": res.get("errors"),
                "upper_bound": res.get("upper_bound"),
                "coverage": res.get("coverage"),
            }
            return out, "jadal.certify"
        except Exception as exc:
            log(f"jadal.certify.pick_threshold failed ({type(exc).__name__}: {exc}); local scan used")
    return (
        local_pick_threshold(confidence, correct, target_error, delta, min_accepted),
        "local_fallback",
    )


def build_report(
    select_records: Sequence[dict],
    report_records: Sequence[dict],
    bins: int,
    certify,
    target_error: float,
    delta: float,
    min_accepted: int,
    select_desc: str,
    report_desc: str,
) -> dict:
    """Threshold on the SELECTION set; accuracy/ECE/curve on the REPORT set.

    The selection certificate (exact-binomial upper bound at level delta)
    comes from the selection set only. Applying the threshold to the report
    set gives observed coverage/error there — reported separately and never
    described as certified.
    """
    rep_conf = [r["p_yes"] for r in report_records]
    rep_y = [r["y"] for r in report_records]
    rep_confidence, rep_correct = _conf_correct(report_records)

    ece_source = "local_fallback"
    ece_value = local_ece(rep_conf, rep_y, bins=bins)
    rc_source = "local_fallback"
    points = local_risk_coverage(rep_confidence, rep_correct)

    if certify is not None:
        try:
            ece_value = _round(float(certify.ece(rep_conf, rep_y, bins=bins)))
            ece_source = "jadal.certify"
        except Exception as exc:
            log(f"jadal.certify.ece failed ({type(exc).__name__}: {exc}); local ECE kept")
        try:
            rc = certify.risk_coverage(rep_confidence, rep_correct)
            points = [
                {"threshold": _round(p["threshold"]), "coverage": _round(p["coverage"]), "error": _round(p["error"])}
                for p in rc
            ]
            rc_source = "jadal.certify"
        except Exception as exc:
            log(f"jadal.certify.risk_coverage failed ({type(exc).__name__}: {exc}); local curve kept")

    n = len(report_records)
    accuracy = _round(sum(rep_correct) / n) if n else 0.0

    selection, threshold_source = ({"threshold": None}, "none")
    try:
        selection, threshold_source = select_threshold(
            select_records, certify, target_error, delta, min_accepted
        )
    except Exception as exc:  # never let a threshold hunt kill the report
        log(f"threshold selection failed: {type(exc).__name__}: {exc}")
        selection = {"threshold": None}

    # observed operating point of the selected threshold on the REPORT set
    applied: dict = {"threshold": selection.get("threshold")}
    if selection.get("threshold") is not None and n:
        tau = float(selection["threshold"])
        acc_idx = [i for i, c in enumerate(rep_confidence) if c >= tau]
        applied["accepted"] = len(acc_idx)
        applied["coverage"] = _round(len(acc_idx) / n)
        applied["error"] = _round(1.0 - sum(rep_correct[i] for i in acc_idx) / len(acc_idx)) if acc_idx else None
        applied["note"] = "observed on the report set; NOT certified (certificate lives on the selection set)"
    else:
        applied.update({"accepted": 0, "coverage": 0.0, "error": None,
                        "note": "no threshold selected; nothing applied"})

    return {
        "created_utc": _utc_now(),
        "tool": "eval/calibration_report.py",
        "selection_set": select_desc,
        "report_set": report_desc,
        "n_select": len(select_records),
        "n": n,
        "n_positive": int(sum(rep_y)),
        "n_negative": int(n - sum(rep_y)),
        "bins": bins,
        "accuracy": accuracy,
        "ece": _round(ece_value),
        "ece_source": ece_source,
        "risk_coverage_source": rc_source,
        "certify_present": certify is not None,
        "label_note": "labels are gemini-2.5-flash auto-labels, NOT human-checked (eval/README.md)",
        "threshold": {
            "target_error": _round(target_error),
            "delta": _round(delta),
            "min_accepted": min_accepted,
            "source": threshold_source,
            "selection": selection,
            "applied_to_report": applied,
        },
        "risk_coverage": points,
    }


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def write_png(path: Path, records: Sequence[dict], bins: int, title_suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = [r["p_yes"] for r in records]
    y = [r["y"] for r in records]
    pred = [int(p >= 0.5) for p in conf]
    correct = [int(pr == int(yi)) for pr, yi in zip(pred, y)]
    confidence = [max(float(p), 1.0 - float(p)) for p in conf]
    points = local_risk_coverage(confidence, correct)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # reliability diagram (binned by predicted-class confidence)
    stats = bin_stats(confidence, correct, bins)
    xs = [s["conf"] for s in stats]
    ys = [s["acc"] for s in stats]
    cs = [s["n"] for s in stats]
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    if xs:
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=40, edgecolors="k", linewidths=0.5, label="bins")
        plt.colorbar(sc, ax=ax, label="n in bin")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("mean predicted-class confidence max(p, 1-p)")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"reliability (ECE={local_ece(conf, y, bins=bins):.4f})")
    ax.legend(loc="upper left", fontsize=8)

    # accuracy / error vs coverage
    ax = axes[1]
    cov = [p["coverage"] for p in points]
    acc = [1.0 - p["error"] for p in points]
    ax.plot(cov, acc, "o-", markersize=3, linewidth=1.2, label="accuracy")
    ax.plot(cov, [p["error"] for p in points], "s--", markersize=3, linewidth=1.0, color="tab:red", label="error")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("coverage (frames with confidence >= threshold)")
    ax.set_ylabel("accuracy / error on accepted")
    ax.set_title("accuracy vs coverage")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(f"Jadal eval calibration{title_suffix}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Accuracy-vs-coverage + ECE report for ego10k predictions.")
    p.add_argument("--predictions", type=Path, default=RESULTS_DIR / "predictions.json",
                   help="predictions file for the REPORT set")
    p.add_argument("--report-split", type=str, default=None, choices=["train", "calibration", "test"],
                   help="restrict the report set to this split (requires --splits)")
    p.add_argument("--split", type=str, default=None, choices=["train", "calibration", "test"],
                   help="deprecated alias for --report-split")
    p.add_argument("--select-predictions", type=Path, default=None,
                   help="predictions file for threshold SELECTION (default: same as --predictions)")
    p.add_argument("--select-split", type=str, default=None, choices=["train", "calibration", "test"],
                   help="restrict the selection set to this split (requires --splits; "
                        "default 'calibration' when --splits is given)")
    p.add_argument("--splits", type=Path, default=None, help="splits.json manifest for split filtering")
    p.add_argument("--json-out", type=Path, default=RESULTS_DIR / "calibration.json")
    p.add_argument("--png-out", type=Path, default=RESULTS_DIR / "calibration.png")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS)
    p.add_argument("--target-error", type=float, default=DEFAULT_TARGET_ERROR)
    p.add_argument("--delta", type=float, default=0.1, help="error level for the exact-binomial certificate")
    p.add_argument("--min-accepted", type=int, default=DEFAULT_MIN_ACCEPTED)
    p.add_argument("--write-example", type=Path, default=None, help="write a schema example predictions file and exit")
    return p.parse_args(list(argv) if argv is not None else None)


def write_example(path: Path) -> None:
    example = {
        "predictions": [
            {"frame_id": "ego4d_000001", "probs": {"yes": 0.82, "no": 0.18}, "label": "yes"},
            {"frame_id": "ego4d_000002", "probs": 0.31, "label": "no"},
            {"frame_id": "ego4d_000003", "probs": [0.6, 0.4], "label": 0},
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example, indent=2))
    log(f"wrote schema example to {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_example is not None:
        write_example(args.write_example)
        return 0
    if not args.predictions.exists():
        raise SystemExit(f"predictions file not found: {args.predictions} (see --write-example)")

    report_split = args.report_split or args.split  # --split is a deprecated alias
    if report_split and args.splits is None:
        raise SystemExit("--report-split/--split requires --splits")
    select_file = args.select_predictions or args.predictions
    if not select_file.exists():
        raise SystemExit(f"selection predictions file not found: {select_file}")
    select_split = args.select_split
    if select_split and args.splits is None:
        raise SystemExit("--select-split requires --splits")
    if args.splits is not None and select_split is None and report_split != "calibration":
        select_split = "calibration"  # the standard: select on calibration …

    records, skipped = load_predictions(args.predictions)
    if not records:
        raise SystemExit("no usable predictions parsed")
    dropped = 0
    if report_split:
        records, dropped = restrict_to_split(records, args.splits, report_split)
        if not records:
            raise SystemExit("no predictions left after restricting to the report split")

    if select_file == args.predictions:
        select_records = records if report_split is None and select_split is None else None
        sel_skipped = 0
        if select_records is None:
            all_records, sel_skipped = load_predictions(select_file)
            if select_split:
                select_records, _ = restrict_to_split(all_records, args.splits, select_split)
            else:
                select_records = all_records
    else:
        select_records, sel_skipped = load_predictions(select_file)
        if select_split:
            if args.splits is None:
                raise SystemExit("--select-split requires --splits")
            select_records, _ = restrict_to_split(select_records, args.splits, select_split)
    if not select_records:
        raise SystemExit("no predictions left in the selection set")

    # refuse any overlap between selection and report frame_ids
    overlap = {r["frame_id"] for r in select_records} & {r["frame_id"] for r in records}
    if overlap:
        raise SystemExit(
            f"refusing: {len(overlap)} frame_id(s) are in BOTH the selection and report sets "
            f"(e.g. {sorted(overlap)[:3]}). Select the threshold on calibration, report on test."
        )

    select_desc = f"{select_file}" + (f"[{select_split}]" if select_split else "[all]")
    report_desc = f"{args.predictions}" + (f"[{report_split}]" if report_split else "[all]")

    certify = _load_certify()
    report = build_report(
        select_records,
        records,
        bins=args.bins,
        certify=certify,
        target_error=args.target_error,
        delta=args.delta,
        min_accepted=args.min_accepted,
        select_desc=select_desc,
        report_desc=report_desc,
    )
    report["input"] = {
        "predictions": str(args.predictions),
        "select_predictions": str(select_file),
        "skipped_records": skipped,
        "selection_skipped_records": sel_skipped,
        "dropped_outside_split": dropped,
        "report_split": report_split,
        "select_split": select_split,
        "splits_manifest": str(args.splits) if args.splits else None,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    title = f" (report={report_desc}, n={report['n']})"
    write_png(args.png_out, records, bins=args.bins, title_suffix=title)

    log(
        f"select={select_desc} n_sel={report['n_select']} report={report_desc} n={report['n']} "
        f"accuracy={report['accuracy']} ece={report['ece']} ({report['ece_source']}) "
        f"threshold={report['threshold']['selection'].get('threshold')} "
        f"({report['threshold']['source']}) applied_cov={report['threshold']['applied_to_report'].get('coverage')}"
    )
    log(f"wrote {args.json_out} and {args.png_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
