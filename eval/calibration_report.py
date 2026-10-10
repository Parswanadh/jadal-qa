"""Jadal QA: calibration report for predictions on the ego10k eval split.

Usage (from the repo root):

    /home/parshu/envs/jadal/bin/python -m eval.calibration_report \
        --predictions eval/results/predictions.json \
        --splits eval/results/splits.json --split test

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
        if len(keys) == 2:  # {"no": .., "yes": ..} spelled differently -> take the larger key? no: ordered
            values = [float(v) for v in probs.values()]
            if len(values) == 2:
                return _clip(values[1])
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
    if isinstance(label, (int, float)):
        if label == 1:
            return 1
        if label == 0:
            return 0
        if 0.0 < float(label) < 1.0:
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
        p_yes = item["probs"].get("p_yes") if isinstance(item.get("probs"), dict) and "p_yes" in item["probs"] else _as_p_yes(item.get("probs", item.get("prob", item.get("p_yes", item.get("score")))))
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


def build_report(records: Sequence[dict], bins: int, certify, target_error: float, min_accepted: int) -> dict:
    conf = [r["p_yes"] for r in records]
    y = [r["y"] for r in records]
    pred = [int(p >= 0.5) for p in conf]
    correct = [int(pr == int(yi)) for pr, yi in zip(pred, y)]
    confidence = [max(float(p), 1.0 - float(p)) for p in conf]

    ece_source = "local_fallback"
    ece_value = local_ece(conf, y, bins=bins)
    rc_source = "local_fallback"
    points = local_risk_coverage(confidence, correct)

    if certify is not None:
        try:
            ece_value = _round(float(certify.ece(conf, y, bins=bins)))
            ece_source = "jadal.certify"
        except Exception as exc:
            log(f"jadal.certify.ece failed ({type(exc).__name__}: {exc}); local ECE kept")
        try:
            rc = certify.risk_coverage(confidence, correct)
            points = [
                {"threshold": _round(p["threshold"]), "coverage": _round(p["coverage"]), "error": _round(p["error"])}
                for p in rc
            ]
            rc_source = "jadal.certify"
        except Exception as exc:
            log(f"jadal.certify.risk_coverage failed ({type(exc).__name__}: {exc}); local curve kept")

    n = len(records)
    accuracy = _round(sum(correct) / n) if n else 0.0
    points = [p for p in points if p["error"] is not None]

    threshold_info: dict = {"target_error": _round(target_error), "min_accepted": min_accepted}
    try:
        threshold_info.update(local_choose_threshold(points, n, target_error, min_accepted, certify))
    except Exception as exc:  # never let a threshold hunt kill the report
        log(f"threshold selection failed: {type(exc).__name__}: {exc}")
        threshold_info["threshold"] = None

    return {
        "created_utc": _utc_now(),
        "tool": "eval/calibration_report.py",
        "n": n,
        "n_positive": int(sum(y)),
        "n_negative": int(n - sum(y)),
        "bins": bins,
        "accuracy": accuracy,
        "ece": _round(ece_value),
        "ece_source": ece_source,
        "risk_coverage_source": rc_source,
        "certify_present": certify is not None,
        "label_note": "labels are gemini-2.5-flash auto-labels, NOT human-checked (eval/README.md)",
        "threshold": threshold_info,
        "risk_coverage": points,
    }


def local_choose_threshold(
    points: list[dict], n: int, target_error: float, min_accepted: int, certify
) -> dict:
    """Pick the accept-threshold with the best coverage at error <= target_error."""
    def _pick(candidates: list[dict]) -> dict:
        best = max(candidates, key=lambda p: (p["coverage"], -p["threshold"]))
        accepted = int(round(best["coverage"] * n))
        return {
            "threshold": best["threshold"],
            "coverage": best["coverage"],
            "error": best["error"],
            "accepted": accepted,
        }

    if certify is not None and hasattr(certify, "pick_threshold"):
        try:
            res = certify.pick_threshold(
                [p["threshold"] for p in points],
                [int(round((1.0 - p["error"]) * n)) for p in points],
                target_error=target_error,
                min_accepted=min_accepted,
            )
            return {"threshold": res.get("threshold"), "from": "jadal.certify.pick_threshold", "raw": res}
        except Exception as exc:
            log(f"jadal.certify.pick_threshold unusable ({type(exc).__name__}: {exc}); local pick used")

    eligible = [p for p in points if p["error"] <= target_error and int(round(p["coverage"] * n)) >= min_accepted]
    if not eligible:
        relaxed = [p for p in points if p["error"] <= target_error]
        source = "local_min_accepted"
        if not relaxed:
            return {"threshold": None, "coverage": 0.0, "error": None, "accepted": 0, "from": "none_eligible"}
        return {**_pick(relaxed), "from": source}
    return {**_pick(eligible), "from": "local_min_accepted"}


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
    p.add_argument("--predictions", type=Path, default=RESULTS_DIR / "predictions.json")
    p.add_argument("--splits", type=Path, default=None, help="splits.json manifest; with --split, restrict the report to that split")
    p.add_argument("--split", type=str, default=None, choices=["train", "calibration", "test"])
    p.add_argument("--json-out", type=Path, default=RESULTS_DIR / "calibration.json")
    p.add_argument("--png-out", type=Path, default=RESULTS_DIR / "calibration.png")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS)
    p.add_argument("--target-error", type=float, default=DEFAULT_TARGET_ERROR)
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

    records, skipped = load_predictions(args.predictions)
    if not records:
        raise SystemExit("no usable predictions parsed")
    dropped = 0
    if args.split:
        if args.splits is None:
            raise SystemExit("--split requires --splits")
        records, dropped = restrict_to_split(records, args.splits, args.split)
        if not records:
            raise SystemExit("no predictions left after restricting to the split")

    certify = _load_certify()
    report = build_report(
        records,
        bins=args.bins,
        certify=certify,
        target_error=args.target_error,
        min_accepted=args.min_accepted,
    )
    report["input"] = {
        "predictions": str(args.predictions),
        "skipped_records": skipped,
        "dropped_outside_split": dropped,
        "split": args.split,
        "splits_manifest": str(args.splits) if args.splits else None,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    title = f" (split={args.split}, n={report['n']})" if args.split else f" (n={report['n']})"
    write_png(args.png_out, records, bins=args.bins, title_suffix=title)

    log(
        f"n={report['n']} skipped={skipped} dropped={dropped} accuracy={report['accuracy']} "
        f"ece={report['ece']} ({report['ece_source']}) curve={report['risk_coverage_source']} "
        f"threshold={report['threshold'].get('threshold')} cov={report['threshold'].get('coverage')}"
    )
    log(f"wrote {args.json_out} and {args.png_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
