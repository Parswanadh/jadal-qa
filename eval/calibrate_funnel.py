"""Two-sided funnel calibration driver (GitHub issue #26, fix round 2).

Sources (exactly one of):
  --records PATH   JSON list of labelled records (or {"records": [...]})
  --synthetic N    deterministic synthetic fixtures (label_source synthetic)
  --from-gold [--gold PATH]
                   join human gold labels (default eval/gold/labels.json)
                   with the manifest segments and compute REAL L0 signals
                   on EXACTLY the gold segments (clip, start_s, end_s) via
                   jadal.signals (needs the signals module importable, else
                   a clear error). With no labels file it exits saying
                   'no human gold labels yet' — everything stays
                   provisional/synthetic until the labelling session lands.

Never pools label sources: calibration runs per source; --source selects
which source's thresholds go into the config file (default: the sole
source, or an error asking for --source). The wall clock is written to a
'sidecar' field excluded from every hash/comparison (M3: funnel output
itself has no clock).

status rule: 'calibrated' ONLY for human_gold with N >= 100 and every
written threshold's CERTIFY bound met; else 'provisional' (source stated).
--write-config from a non-human source refuses to set accept_conf unless
--allow-provisional is given (then status stays 'provisional' with the
label source stated).

Usage:
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        eval/calibrate_funnel.py --synthetic 400 --seed 7
    PYTHONNOUSERSITE=1 /home/parshu/envs/jadal/bin/python \\
        eval/calibrate_funnel.py --from-gold --source human_gold --write-config
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jadal import funnel  # noqa: E402

RESULTS_PATH = ROOT / "eval" / "results" / "funnel_calibration.json"
GOLD_RECORDS_PATH = ROOT / "eval" / "results" / "funnel_gold_records.json"
CONFIG_PATH = ROOT / "configs" / "funnel_thresholds.json"
MANIFEST_PATH = ROOT / "eval" / "gold" / "segments.json"
LABELS_PATH = ROOT / "eval" / "gold" / "labels.json"
CUTS_DIR = ROOT / "data_runtime" / "gold_cuts"


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


_WEARER_RE = re.compile(r"User_(\d+)")


def wearers_from_path(path: str) -> list[str]:
    """Wearer ids parsed from a clip path (e.g. User_15_..._480.mp4).

    IndEgo clip names carry ``User_<id>`` tokens (a clip can show several
    wearers, e.g. ``User_1_User_15_...``). Returns [] when the name
    carries no wearer info (=> provenance says 'unknown', honestly).
    """
    return sorted({match.group(1) for match in _WEARER_RE.finditer(path or "")},
                  key=int)


def cut_segment(src: str, start_s: float, end_s: float, out_path: Path,
                recut: bool = False) -> Path:
    """Deterministically cut [start_s, end_s) to out_path (mp4v re-encode).

    Frame indices round(start*fps)..round(end*fps)-1; source only read.
    Cached: existing cuts are reused unless recut=True.
    """
    if out_path.is_file() and not recut:
        return out_path
    import cv2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip (read-only check): {src}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
        i0, i1 = int(round(start_s * fps)), int(round(end_s * fps))
        if i1 <= i0:
            raise ValueError(f"empty window [{start_s}, {end_s}) in {src}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read {src}")
        height, width = frame.shape[:2]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open writer for {out_path}")
        try:
            index = 0
            while True:
                if i0 <= index < i1:
                    writer.write(frame)
                index += 1
                if index >= i1:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
        finally:
            writer.release()
    finally:
        cap.release()
    return out_path


def _real_signals_fn():
    try:
        from jadal import signals as signals_mod
    except ImportError as exc:
        raise RuntimeError(
            "jadal.signals is not importable in this worktree, so gold "
            "records cannot be built here. Run --from-gold where the "
            "signals module is available."
        ) from exc
    return signals_mod


def build_gold_records(manifest: dict, labels: dict, signals_fn=None,
                       cuts_dir: Path | None = CUTS_DIR,
                       recut: bool = False, cut_fn=None) -> list[dict]:
    """Join gold labels with manifest segments + real L0 signals.

    For every LABELED manifest segment (manifest order): cut exactly
    [start_s, end_s), compute signals with ``signals_fn(cut_path, segment)``
    (defaults to the real ``jadal.signals.compute_signals``), add
    ``duration_s`` from the manifest, resolve near-duplicates by exact
    phash match in manifest order (conservative subset of the pipeline's
    Hamming lookup — documented). Unlabeled segments are skipped (said in
    the returned stats). decisions are None (L1 not built yet). ``cut_fn``
    (same signature as :func:`cut_segment`) is injectable for tests.
    """
    if signals_fn is None:
        signals_mod = _real_signals_fn()
        signals_fn = lambda cut, seg: signals_mod.compute_signals(str(cut))
    if cut_fn is None:
        cut_fn = cut_segment
    latest = labels.get("latest", {}) if isinstance(labels, dict) else {}
    segments = manifest.get("segments", [])
    records: list[dict] = []
    phash_index: dict[str, str] = {}
    skipped = 0
    for segment in segments:
        segment_id = segment["segment_id"]
        entry = latest.get(segment_id)
        if entry is None:
            skipped += 1
            continue
        verdict = entry.get("verdict")
        if verdict not in ("good", "bad"):
            raise ValueError(f"segment {segment_id}: verdict must be "
                             f"good|bad, got {verdict!r}")
        reason = entry.get("reason")
        if verdict == "bad" and reason not in funnel.REASON_ORDER:
            raise ValueError(f"segment {segment_id}: bad verdict needs a "
                             f"valid reason code, got {reason!r}")
        cut = cut_fn(segment["path"], float(segment["start_s"]),
                     float(segment["end_s"]),
                     (cuts_dir or CUTS_DIR) / f"{segment_id}.mp4",
                     recut=recut)
        signals = dict(signals_fn(str(cut), segment))
        signals["duration_s"] = round(
            float(segment["end_s"]) - float(segment["start_s"]), 1)
        phash = signals.get("phash")
        if phash:
            if phash in phash_index:
                signals["dup_of"] = phash_index[phash]
            else:
                phash_index[phash] = segment_id
        records.append({
            "segment_id": segment_id,
            "signals": signals,
            "decisions": None,
            "label": verdict,
            "reason_gold": reason if verdict == "bad" else None,
            "label_source": "human_gold",
            "group": segment.get("clip"),
            # Provenance (audit L3): wearers parsed from the clip path,
            # labeller id when the gold file carries one (the current
            # label tool records none => 'unknown', said plainly).
            "wearer": wearers_from_path(segment.get("path", "")) or None,
            "labeller": entry.get("labeller") or (
                labels.get("labeller") if isinstance(labels, dict) else None),
        })
    print(f"gold join: {len(records)} labelled, {skipped} unlabeled skipped")
    return records


def written_bounds_ok(src_res: dict, targets: dict) -> tuple[bool, list[str]]:
    """Every WRITTEN threshold's CERTIFY bound met its target?"""
    notes: list[str] = []
    ok = True
    accept = src_res["accept"]
    if accept["threshold"] is not None:
        bound = accept["certify"].get("upper_bound")
        if bound is None or bound > targets["target_alpha"]:
            ok = False
            notes.append(f"accept certify bound {bound} > {targets['target_alpha']}")
    for key, entry in src_res["reject_rules"].items():
        if not entry["disabled"]:
            bound = entry["certify"].get("upper_bound")
            if bound is None or bound > targets["target_beta"]:
                ok = False
                notes.append(f"{key} certify bound {bound} > {targets['target_beta']}")
    return ok, notes


def apply_to_config(result: dict, source: str, path: Path,
                    allow_provisional: bool = False) -> dict:
    """Merge one source's calibrated thresholds into the config file.

    Provisional-accept safety (round 2, audit N1): ``accept_conf`` is
    written from a non-human source ONLY when ``allow_provisional`` is
    passed (``--allow-provisional``); otherwise the write keeps
    ``accept_conf=None`` (auto-accept stays disabled) and says so in the
    calibration note. Human gold always writes fully; status is
    'calibrated' only for human_gold N>=100 with every CERTIFY bound met,
    else 'provisional' with the label source stated.
    """
    if source not in result["per_source"]:
        raise ValueError(f"source {source!r} not in calibration result "
                         f"(have {sorted(result['per_source'])})")
    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    src_res = result["per_source"][source]
    meta = result["meta"]
    n = meta["by_source"][source]
    updates = dict(result["threshold_updates"][source])
    provisional_write = source != "human_gold" and not allow_provisional
    if provisional_write:
        updates["accept_conf"] = None
    for key, value in updates.items():
        config[key] = value
    targets = meta["targets"]
    bounds_ok, bound_notes = written_bounds_ok(src_res, targets)
    if source == "human_gold" and n >= 100 and bounds_ok:
        status = "calibrated"
        note = (f"Calibrated on {n} human gold labels; every written "
                f"threshold met its CERTIFY bound.")
    else:
        status = "provisional"
        reasons = [f"label_source={source}"]
        if source != "human_gold":
            reasons.append("not human gold")
        if n < 100:
            reasons.append(f"N={n} < 100")
        if not bounds_ok:
            reasons.extend(bound_notes)
        note = "Provisional (" + "; ".join(reasons) + ")."
    disabled = [k for k, e in src_res["reject_rules"].items()
                if e["disabled"]]
    stopped = [k for k, e in src_res["reject_rules"].items()
               if e["disabled"] and e.get("select_candidate") is not None]
    config["calibration"] = {
        "status": status,
        "label_source": source,
        "n": n,
        "n_select": src_res["n_select"],
        "n_certify": src_res["n_certify"],
        "targets": targets,
        "seed": meta["seed"],
        "accept_certify": src_res["accept"]["certify"],
        "union_certify": src_res["union"],
        "disabled_rules": disabled,
        "note": note + (" Bounds carry only the certificate of their label "
                        "source. Per-rule bounds do not bound the union; "
                        "see union_certify.")
        + (f" Certification stopped {len(stopped)} bad rule(s) "
           f"({', '.join(sorted(stopped))}): certified on SELECT but "
           f"rejected on held-out clips." if stopped else "")
        + (f" accept_conf forced to null: non-human source without "
           f"--allow-provisional (auto-accept stays disabled)."
           if provisional_write else ""),
    }
    config["sha256"] = funnel.thresholds_hash(config)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    funnel.load_thresholds(path)  # raises if the hash is wrong
    return config


def markdown_tables(result: dict) -> str:
    """Render accept/reject/union tables from the result JSON (for the report)."""
    lines = []
    meta = result["meta"]
    lines.append(f"records={meta['n_total']} sources={meta['by_source']} "
                 f"seed={meta['seed']} "
                 f"bonferroni={meta['targets'].get('bonferroni', True)} "
                 f"delta_per_rule={meta.get('delta_per_rule')}")
    for source, src in result["per_source"].items():
        prov = src.get("provenance", {})
        lines.append(f"\n## source={source} "
                     f"(N={meta['by_source'][source]}, "
                     f"select={src['n_select']} "
                     f"({src.get('n_select_clips', '?')} clips), "
                     f"certify={src['n_certify']} "
                     f"({src.get('n_certify_clips', '?')} clips); "
                     f"n_segments={prov.get('n_segments')}, "
                     f"n_clips={prov.get('n_clips')}, "
                     f"n_wearers={prov.get('n_wearers')}, "
                     f"n_labellers={prov.get('n_labellers')}, "
                     f"label_source={prov.get('label_source')}, "
                     f"delta_per_rule={prov.get('delta_per_rule')})")
        accept = src["accept"]
        lines.append("| side | threshold | certify fired | certify errors | "
                     "certify UCB | gate |")
        lines.append("|---|---|---|---|---|---|")
        cert = accept["certify"]
        lines.append(f"| accept | {accept['threshold']} | {cert['fired']} | "
                     f"{cert['errors']} | {cert['upper_bound']} | "
                     f"{accept.get('gate', 'WRITTEN')} |")
        lines.append("| rule | threshold | disabled | certify fired | "
                     "certify errors | certify UCB |")
        lines.append("|---|---|---|---|---|---|")
        for key, entry in src["reject_rules"].items():
            cert = entry["certify"]
            lines.append(f"| {key} | {entry['threshold']} | "
                         f"{entry['disabled']} | {cert['fired']} | "
                         f"{cert['errors']} | {cert['upper_bound']} |")
        for key, entry in src["reject_rules"].items():
            if entry["disabled"] and entry.get("select_candidate") is not None:
                lines.append(f"STOPPED ({key}): certification stopped a bad "
                             f"rule — {entry.get('reason')}")
            elif entry["disabled"]:
                lines.append(f"DISABLED ({key}): {entry.get('reason')}")
        union = src["union"]
        lines.append(f"union CERTIFY: rejected={union['fired']} "
                     f"errors={union['errors']} UCB={union['upper_bound']} "
                     f"(n_certify={union['n_certify']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--records", type=Path,
                       help="JSON file with a list of labelled records")
    group.add_argument("--synthetic", type=int, metavar="N",
                       help="generate N deterministic synthetic records")
    group.add_argument("--from-gold", action="store_true",
                       help="build records from eval/gold labels + real L0")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--gold", type=Path, default=None,
                        help="human gold labels JSON (default "
                             "eval/gold/labels.json); the moment the "
                             "labelling session produces it, --from-gold "
                             "uses it")
    parser.add_argument("--labels", type=Path, default=None,
                        help="deprecated alias for --gold")
    parser.add_argument("--cuts-dir", type=Path, default=CUTS_DIR)
    parser.add_argument("--recut", action="store_true")
    parser.add_argument("--save-records", type=Path, default=None)
    parser.add_argument("--source", type=str, default=None,
                        help="label source whose thresholds update the config")
    parser.add_argument("--allow-provisional", action="store_true",
                        help="with --write-config from a non-human source, "
                             "write its accept_conf too (status stays "
                             "provisional, source stated); without it, "
                             "accept_conf is forced to null")
    parser.add_argument("--no-bonferroni", action="store_true",
                        help="test each reject rule at full delta (round-1 "
                             "behaviour; NOT family-wise — say so)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--min-support", type=int, default=10)
    parser.add_argument("--min-accepted", type=int, default=30)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--write-config", action="store_true",
                        help="also merge one source's result into thresholds")
    parser.add_argument("--markdown", action="store_true",
                        help="print report tables generated from the JSON")
    args = parser.parse_args(argv)

    if args.records:
        with open(args.records, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload if isinstance(payload, list) \
            else payload.get("records", [])
        provenance = f"records-file:{args.records}"
    elif args.synthetic is not None:
        records = synthetic_records(args.synthetic, args.seed)
        provenance = "synthetic-fixtures"
    else:
        with open(args.manifest, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        labels_path = args.gold or args.labels or LABELS_PATH
        if not labels_path.is_file():
            raise SystemExit(
                f"no human gold labels yet at {labels_path}; run the "
                f"labelling session first (tools/gold_label/label_server.py), "
                f"then rerun --from-gold [--gold PATH]. Everything stays "
                f"provisional/synthetic until then.")
        with open(labels_path, "r", encoding="utf-8") as fh:
            labels = json.load(fh)
        records = build_gold_records(manifest, labels, signals_fn=None,
                                     cuts_dir=args.cuts_dir, recut=args.recut)
        provenance = f"gold-join:{labels_path}"
        if args.save_records is None:
            args.save_records = GOLD_RECORDS_PATH
    if args.save_records is not None:
        args.save_records.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_records, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.save_records} ({len(records)} records)")

    result = funnel.calibrate_two_sided(records, {
        "target_alpha": args.alpha, "target_beta": args.beta,
        "delta": args.delta, "min_support": args.min_support,
        "min_accepted": args.min_accepted, "seed": args.seed,
        "bonferroni": not args.no_bonferroni})
    result["meta"]["provenance"] = provenance
    # M3: the wall clock lives in a sidecar excluded from every hash and
    # comparison (see funnel.calibration_content_hash), never in meta.
    result["sidecar"] = {
        "produced_utc": datetime.now(timezone.utc).isoformat(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(markdown_tables(result))
    print(f"wrote {args.out}")
    if args.write_config:
        sources = sorted(result["per_source"])
        source = args.source or (sources[0] if len(sources) == 1 else None)
        if source is None:
            raise SystemExit(
                f"multiple sources {sources}: pass --source to choose which "
                f"source updates the config (never pooled)")
        apply_to_config(result, source, args.config,
                        allow_provisional=args.allow_provisional)
        print(f"updated {args.config} from source={source} "
              f"(allow_provisional={args.allow_provisional})")
    else:
        print("(config file untouched: pass --write-config to update it)")
    if args.markdown:
        print("---- markdown ----")
        print(markdown_tables(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
